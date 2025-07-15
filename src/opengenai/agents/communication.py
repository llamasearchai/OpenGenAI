"""
Agent Communication System for OpenGenAI
Handles inter-agent messaging, coordination, and event distribution.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from opengenai.core.exceptions import (
    AgentError,
    ResourceNotFoundError,
    ValidationError,
)
from opengenai.core.logging import get_logger
from opengenai.core.types import AgentMessage, MessageType

logger = get_logger(__name__)


class DeliveryStatus(str, Enum):
    """Message delivery status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MessageDeliveryInfo:
    """Information about message delivery."""

    message_id: str
    sender_id: str
    recipient_id: str
    status: DeliveryStatus
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None
    failed_at: datetime | None = None
    error_message: str | None = None


class MessageChannel:
    """Channel for message delivery between agents."""

    def __init__(self, name: str, max_size: int = 1000):
        """Initialize message channel."""
        self.name = name
        self.max_size = max_size
        self.queue: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=max_size)
        self.subscribers: set[str] = set()
        self.created_at = datetime.now(UTC)
        self.message_count = 0
        self.active = True

    async def send_message(self, message: AgentMessage) -> None:
        """Send message to channel."""
        if not self.active:
            raise AgentError(
                f"Channel '{self.name}' is not active",
                agent_id=message.sender_id,
                agent_name=self.name,
            )

        try:
            await self.queue.put(message)
            self.message_count += 1
            logger.debug(
                "Message sent to channel",
                channel=self.name,
                message_id=message.id,
                sender=message.sender_id,
            )
        except asyncio.QueueFull:
            raise AgentError(
                f"Channel '{self.name}' is full",
                agent_id=message.sender_id,
                agent_name=self.name,
            )

    async def receive_message(self, timeout: float | None = None) -> AgentMessage | None:
        """Receive message from channel."""
        if not self.active:
            return None

        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def subscribe(self, agent_id: str) -> None:
        """Subscribe agent to channel."""
        self.subscribers.add(agent_id)
        logger.debug(
            "Agent subscribed to channel",
            agent_id=agent_id,
            channel=self.name,
        )

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe agent from channel."""
        self.subscribers.discard(agent_id)
        logger.debug(
            "Agent unsubscribed from channel",
            agent_id=agent_id,
            channel=self.name,
        )

    def is_subscribed(self, agent_id: str) -> bool:
        """Check if agent is subscribed to channel."""
        return agent_id in self.subscribers

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self.name,
            "active": self.active,
            "queue_size": self.queue.qsize(),
            "max_size": self.max_size,
            "subscribers": len(self.subscribers),
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat(),
        }

    def close(self) -> None:
        """Close the channel."""
        self.active = False
        self.subscribers.clear()
        logger.info("Channel closed", channel=self.name)


class MessageBroker:
    """Message broker for inter-agent communication.

    The original implementation started background asyncio tasks from the
    constructor. Instantiating the broker in a **non-async** context (for
    example, inside synchronous test setup code) causes a
    `RuntimeError: no running event loop` when `asyncio.create_task(...)` is
    executed.  To make the broker safe to construct in synchronous code we now
    defer creation of background tasks until an event loop is *actually*
    running via the new `start()` coroutine.  The broker therefore follows a
    simple lifecycle:

    1. `__init__()`   – lightweight, no interaction with the event loop.
    2. `await start()` – spawn background workers, mark broker active.
    3. `await stop()`  – cancel workers, flush/close channels, mark inactive.

    For compatibility with existing code paths that invoked `shutdown()`, we
    keep `shutdown` as an alias to `stop`.  Likewise, `AgentManager` expects a
    `stop()` coroutine, so we expose both names.
    """

    def __init__(self) -> None:  # pragma: no cover – trivial
        """Create a *passive* broker instance (no running tasks yet)."""

        # Core data structures
        self.channels: dict[str, MessageChannel] = {}
        self.agent_queues: dict[str, asyncio.Queue[AgentMessage]] = {}
        self.delivery_info: dict[str, MessageDeliveryInfo] = {}
        self.message_handlers: dict[str, dict[MessageType, Callable]] = {}
        self.event_listeners: dict[str, list[Callable]] = {}
        self.routing_rules: dict[str, Callable[[AgentMessage], str]] = {}

        # Lifecycle flags / background worker handles – populated in start()
        self.active: bool = False
        self._delivery_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Default channels are always available irrespective of the running
        # state.
        self._create_default_channels()

    # ---------------------------------------------------------------------
    # Lifecycle management
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        """Activate the broker and launch background maintenance workers."""

        if self.active:  # Idempotent – do nothing if already running
            return

        self.active = True

        loop = asyncio.get_running_loop()
        self._delivery_task = loop.create_task(
            self._delivery_worker(), name="broker-delivery-worker"
        )
        self._cleanup_task = loop.create_task(self._cleanup_worker(), name="broker-cleanup-worker")

        logger.info("Message broker started")

    async def stop(self) -> None:  # noqa: D401 – imperative mood preferred
        """Gracefully shut down the broker and its background workers."""

        if not self.active:
            # Not running – nothing to do
            return

        self.active = False

        # Cancel background workers in parallel
        for task in (self._delivery_task, self._cleanup_task):
            if task is not None:
                task.cancel()

        # Await their termination, swallowing *only* `CancelledError`.
        for task in (self._delivery_task, self._cleanup_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all channels
        for channel in self.channels.values():
            channel.close()

        logger.info("Message broker stopped")

    # Backwards-compatibility aliases -------------------------------------------------

    # Existing code (e.g. FastAPI lifespan handler) calls `shutdown()`.  We keep it
    # as a thin wrapper around `stop()` to avoid touching those call-sites.
    shutdown = stop  # type: ignore – intentional alias

    def _create_default_channels(self) -> None:
        """Create default message channels."""
        default_channels = [
            "broadcast",
            "system",
            "alerts",
            "coordination",
            "monitoring",
        ]

        for channel_name in default_channels:
            self.create_channel(channel_name)

    def create_channel(self, name: str, max_size: int = 1000) -> MessageChannel:
        """Create a new message channel."""
        if name in self.channels:
            raise ResourceNotFoundError(
                f"Channel '{name}' already exists",
                resource_type="channel",
                resource_id=name,
            )

        channel = MessageChannel(name, max_size)
        self.channels[name] = channel

        logger.info("Channel created", channel=name, max_size=max_size)
        return channel

    def get_channel(self, name: str) -> MessageChannel:
        """Get message channel by name."""
        if name not in self.channels:
            raise ResourceNotFoundError(
                f"Channel '{name}' not found",
                resource_type="channel",
                resource_id=name,
            )

        return self.channels[name]

    def delete_channel(self, name: str) -> None:
        """Delete a message channel."""
        if name not in self.channels:
            raise ResourceNotFoundError(
                f"Channel '{name}' not found",
                resource_type="channel",
                resource_id=name,
            )

        channel = self.channels[name]
        channel.close()
        del self.channels[name]

        logger.info("Channel deleted", channel=name)

    def register_agent(self, agent_id: str) -> None:
        """Register an agent with the message broker."""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue()
            self.message_handlers[agent_id] = {}

            logger.info("Agent registered with broker", agent_id=agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message broker."""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]

        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]

        # Unsubscribe from all channels
        for channel in self.channels.values():
            channel.unsubscribe(agent_id)

        logger.info("Agent unregistered from broker", agent_id=agent_id)

    def subscribe_to_channel(self, agent_id: str, channel_name: str) -> None:
        """Subscribe agent to a channel."""
        channel = self.get_channel(channel_name)
        channel.subscribe(agent_id)

        # Register agent if not already registered
        if agent_id not in self.agent_queues:
            self.register_agent(agent_id)

    def unsubscribe_from_channel(self, agent_id: str, channel_name: str) -> None:
        """Unsubscribe agent from a channel."""
        channel = self.get_channel(channel_name)
        channel.unsubscribe(agent_id)

    async def send_message(
        self,
        sender_id: str,
        recipient_id: str | None,
        content: Any,
        message_type: MessageType = MessageType.REQUEST,
        channel: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send message to agent or channel."""
        if not self.active:
            raise AgentError("Message broker is not active")

        # Create message
        message = AgentMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        # Apply routing rules
        if sender_id in self.routing_rules:
            try:
                route_channel = self.routing_rules[sender_id](message)
                if route_channel:
                    channel = route_channel
            except Exception as e:
                logger.warning(
                    "Routing rule failed",
                    sender_id=sender_id,
                    error=str(e),
                )

        # Create delivery info
        delivery_info = MessageDeliveryInfo(
            message_id=message.id,
            sender_id=sender_id,
            recipient_id=recipient_id or channel or "unknown",
            status=DeliveryStatus.PENDING,
        )
        self.delivery_info[message.id] = delivery_info

        try:
            if channel:
                # Send to channel
                await self._send_to_channel(message, channel)
            elif recipient_id:
                # Send to specific agent
                await self._send_to_agent(message, recipient_id)
            else:
                raise ValidationError(
                    "Either recipient_id or channel must be specified",
                    field="recipient_id",
                )

            delivery_info.status = DeliveryStatus.DELIVERED
            delivery_info.delivered_at = datetime.now(UTC)

            logger.info(
                "Message sent",
                message_id=message.id,
                sender=sender_id,
                recipient=recipient_id,
                channel=channel,
                type=message_type,
            )

            return message.id

        except Exception as e:
            delivery_info.status = DeliveryStatus.FAILED
            delivery_info.failed_at = datetime.now(UTC)
            delivery_info.error_message = str(e)

            logger.warning(
                "Message delivery failed",
                message_id=message.id,
                sender=sender_id,
                recipient=recipient_id,
                error=str(e),
            )

            raise AgentError(
                f"Failed to send message to {recipient_id}: {str(e)}",
                agent_id=sender_id,
                agent_name=sender_id,
            ) from e

    async def _send_to_channel(self, message: AgentMessage, channel_name: str) -> None:
        """Send message to channel."""
        channel = self.get_channel(channel_name)
        await channel.send_message(message)

        # Deliver to all subscribers
        for subscriber_id in channel.subscribers:
            try:
                await self._send_to_agent(message, subscriber_id)
            except Exception as e:
                logger.warning(
                    "Failed to deliver channel message to subscriber",
                    message_id=message.id,
                    channel=channel_name,
                    subscriber=subscriber_id,
                    error=str(e),
                )

    async def _send_to_agent(self, message: AgentMessage, agent_id: str) -> None:
        """Send message to specific agent."""
        if agent_id not in self.agent_queues:
            raise ResourceNotFoundError(
                f"Agent '{agent_id}' not registered",
                resource_type="agent",
                resource_id=agent_id,
            )

        try:
            await self.agent_queues[agent_id].put(message)

            # Trigger event listeners
            if agent_id in self.event_listeners:
                for listener in self.event_listeners[agent_id]:
                    try:
                        await listener(message)
                    except Exception as e:
                        logger.warning(
                            "Event listener failed",
                            agent_id=agent_id,
                            message_id=message.id,
                            error=str(e),
                        )
        except asyncio.QueueFull:
            raise AgentError(
                f"Agent '{agent_id}' message queue is full",
                agent_id=agent_id,
                agent_name=agent_id,
            )

    async def receive_message(
        self,
        agent_id: str,
        timeout: float | None = None,
    ) -> AgentMessage | None:
        """Receive message for agent."""
        if agent_id not in self.agent_queues:
            raise ResourceNotFoundError(
                f"Agent '{agent_id}' not registered",
                resource_type="agent",
                resource_id=agent_id,
            )

        try:
            message = await asyncio.wait_for(
                self.agent_queues[agent_id].get(),
                timeout=timeout,
            )

            # Handle message with registered handler
            if agent_id in self.message_handlers:
                handler = self.message_handlers[agent_id].get(message.type)
                if handler:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(
                            "Message handler failed",
                            agent_id=agent_id,
                            message_id=message.id,
                            error=str(e),
                        )

            return message

        except TimeoutError:
            return None

    def register_message_handler(
        self,
        agent_id: str,
        message_type: MessageType,
        handler: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """Register message handler for agent."""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}

        self.message_handlers[agent_id][message_type] = handler

        logger.debug(
            "Message handler registered",
            agent_id=agent_id,
            message_type=message_type,
        )

    def register_event_listener(
        self,
        agent_id: str,
        listener: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """Register event listener for agent."""
        if agent_id not in self.event_listeners:
            self.event_listeners[agent_id] = []

        self.event_listeners[agent_id].append(listener)

        logger.debug("Event listener registered", agent_id=agent_id)

    def register_routing_rule(
        self,
        agent_id: str,
        rule: Callable[[AgentMessage], str],
    ) -> None:
        """Register routing rule for agent."""
        self.routing_rules[agent_id] = rule

        logger.debug("Routing rule registered", agent_id=agent_id)

    async def broadcast_message(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.BROADCAST,
        exclude_agents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Broadcast message to all registered agents."""
        exclude_agents = exclude_agents or []
        message_ids = []

        for agent_id in self.agent_queues:
            if agent_id == sender_id or agent_id in exclude_agents:
                continue

            try:
                message_id = await self.send_message(
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    content=content,
                    message_type=message_type,
                    metadata=metadata,
                )
                message_ids.append(message_id)
            except Exception as e:
                logger.warning(
                    "Failed to broadcast to agent",
                    agent_id=agent_id,
                    error=str(e),
                )

        logger.info(
            "Message broadcast",
            sender=sender_id,
            recipients=len(message_ids),
            type=message_type,
        )

        return message_ids

    def get_delivery_status(self, message_id: str) -> MessageDeliveryInfo | None:
        """Get message delivery status."""
        return self.delivery_info.get(message_id)

    def get_broker_stats(self) -> dict[str, Any]:
        """Get broker statistics."""
        return {
            "active": self.active,
            "registered_agents": len(self.agent_queues),
            "channels": len(self.channels),
            "pending_messages": sum(queue.qsize() for queue in self.agent_queues.values()),
            "total_deliveries": len(self.delivery_info),
            "failed_deliveries": sum(
                1 for info in self.delivery_info.values() if info.status == DeliveryStatus.FAILED
            ),
            "channel_stats": {name: channel.get_stats() for name, channel in self.channels.items()},
        }

    async def wait_for_response(
        self,
        agent_id: str,
        correlation_id: str,
        timeout: float = 30.0,
    ) -> AgentMessage | None:
        """Wait for response message with correlation ID."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            message = await self.receive_message(agent_id, timeout=1.0)
            if message and message.correlation_id == correlation_id:
                return message

        return None

    async def request_response(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage | None:
        """Send request and wait for response."""
        correlation_id = str(uuid.uuid4())

        # Send request
        await self.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=MessageType.REQUEST,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        # Wait for response
        return await self.wait_for_response(sender_id, correlation_id, timeout)

    async def _delivery_worker(self) -> None:
        """Background worker for handling delivery retries."""
        while self.active:
            try:
                # Check for failed deliveries that need retry
                current_time = datetime.now(UTC)

                for message_id, delivery_info in list(self.delivery_info.items()):
                    if (
                        delivery_info.status == DeliveryStatus.FAILED
                        and delivery_info.attempts < delivery_info.max_attempts
                    ):

                        # Retry delivery
                        delivery_info.attempts += 1
                        delivery_info.status = DeliveryStatus.PENDING

                        logger.info(
                            "Retrying message delivery",
                            message_id=message_id,
                            attempt=delivery_info.attempts,
                        )

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Delivery worker error", error=str(e))
                await asyncio.sleep(10)

    async def _cleanup_worker(self) -> None:
        """Background worker for cleanup tasks."""
        while self.active:
            try:
                # Clean up old delivery info
                current_time = datetime.now(UTC)
                cutoff_time = current_time.timestamp() - 3600  # 1 hour ago

                expired_messages = [
                    message_id
                    for message_id, delivery_info in self.delivery_info.items()
                    if delivery_info.created_at.timestamp() < cutoff_time
                ]

                for message_id in expired_messages:
                    del self.delivery_info[message_id]

                if expired_messages:
                    logger.debug(
                        "Cleaned up old delivery info",
                        count=len(expired_messages),
                    )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error("Cleanup worker error", error=str(e))
                await asyncio.sleep(60)


class MessageFilter:
    """Filter for message routing and processing."""

    def __init__(self, name: str):
        """Initialize message filter."""
        self.name = name
        self.rules: list[Callable[[AgentMessage], bool]] = []

    def add_rule(self, rule: Callable[[AgentMessage], bool]) -> None:
        """Add filtering rule."""
        self.rules.append(rule)

    def matches(self, message: AgentMessage) -> bool:
        """Check if message matches filter."""
        return all(rule(message) for rule in self.rules)

    @staticmethod
    def by_sender(sender_id: str) -> Callable[[AgentMessage], bool]:
        """Create filter by sender."""
        return lambda msg: msg.sender_id == sender_id

    @staticmethod
    def by_type(message_type: MessageType) -> Callable[[AgentMessage], bool]:
        """Create filter by message type."""
        return lambda msg: msg.type == message_type

    @staticmethod
    def by_content_key(key: str) -> Callable[[AgentMessage], bool]:
        """Create filter by content key."""
        return lambda msg: isinstance(msg.content, dict) and key in msg.content

    @staticmethod
    def by_metadata_key(key: str) -> Callable[[AgentMessage], bool]:
        """Create filter by metadata key."""
        return lambda msg: key in msg.metadata


# Global message broker instance
_global_broker: MessageBroker | None = None


def get_message_broker() -> MessageBroker:
    """Get the global message broker."""
    global _global_broker
    if _global_broker is None:
        _global_broker = MessageBroker()
    return _global_broker


async def send_message(
    sender_id: str,
    recipient_id: str | None,
    content: Any,
    message_type: MessageType = MessageType.REQUEST,
    channel: str | None = None,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Send message using global broker."""
    broker = get_message_broker()
    return await broker.send_message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        content=content,
        message_type=message_type,
        channel=channel,
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def receive_message(
    agent_id: str,
    timeout: float | None = None,
) -> AgentMessage | None:
    """Receive message using global broker."""
    broker = get_message_broker()
    return await broker.receive_message(agent_id, timeout)


def register_agent(agent_id: str) -> None:
    """Register agent with global broker."""
    broker = get_message_broker()
    broker.register_agent(agent_id)
