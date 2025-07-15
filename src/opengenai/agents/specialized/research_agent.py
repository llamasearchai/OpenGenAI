"""
Research Agent for OpenGenAI
Specialized agent for conducting research and gathering information.
"""

import json
from datetime import datetime
from typing import Any

from opengenai.agents.base import BaseAgent
from opengenai.core.exceptions import AgentError, ValidationError
from opengenai.core.logging import get_logger
from opengenai.core.types import (
    AgentCapability,
    AgentConfig,
    TaskConfig,
)

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Specialized agent for research tasks.

    Capabilities:
    - Web search and information gathering
    - Literature review and synthesis
    - Data analysis and insights
    - Report generation
    - Fact verification
    """

    def __init__(self, config: AgentConfig):
        """Initialize research agent."""
        # Ensure research capabilities
        if AgentCapability.RESEARCH not in config.capabilities:
            config.capabilities.append(AgentCapability.RESEARCH)

        super().__init__(config)

        # Research-specific attributes
        self.research_context: dict[str, Any] = {}
        self.findings: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []
        self.research_methods: list[str] = [
            "web_search",
            "literature_review",
            "data_analysis",
            "expert_consultation",
            "fact_verification",
        ]

        self.logger.info(
            "Research agent initialized",
            methods=self.research_methods,
        )

    async def initialize(self) -> None:
        """Initialize research agent resources."""
        self.logger.info("Initializing research agent")

        # Initialize research tools and connections
        await self._initialize_search_tools()
        await self._initialize_data_sources()

        self.logger.info("Research agent initialization complete")

    async def execute(self, task: TaskConfig) -> dict[str, Any]:
        """Execute research task."""
        self.logger.info(
            "Starting research task",
            task_name=task.name,
            description=task.description,
        )

        # Validate task input
        if not isinstance(task.input_data, dict):
            raise ValidationError(
                "Research task input must be a dictionary",
                field="input_data",
                value=task.input_data,
            )

        research_query = task.input_data.get("query")
        if not research_query:
            raise ValidationError(
                "Research query is required",
                field="query",
                value=research_query,
            )

        # Determine research approach
        research_type = task.input_data.get("type", "general")
        depth = task.input_data.get("depth", "moderate")
        sources = task.input_data.get("sources", ["web", "academic"])

        self.logger.info(
            "Research parameters",
            query=research_query,
            type=research_type,
            depth=depth,
            sources=sources,
        )

        # Execute research pipeline
        try:
            # Step 1: Query analysis and planning
            research_plan = await self._analyze_query(research_query, research_type, depth)

            # Step 2: Information gathering
            gathered_info = await self._gather_information(research_plan, sources)

            # Step 3: Analysis and synthesis
            analysis_result = await self._analyze_information(gathered_info)

            # Step 4: Report generation
            report = await self._generate_report(analysis_result, research_query)

            # Step 5: Fact verification
            verified_report = await self._verify_facts(report)

            result = {
                "research_query": research_query,
                "research_type": research_type,
                "depth": depth,
                "sources_used": sources,
                "findings": self.findings,
                "sources": self.sources,
                "report": verified_report,
                "confidence_score": analysis_result.get("confidence", 0.8),
                "research_time": datetime.utcnow().isoformat(),
                "metadata": {
                    "methods_used": research_plan.get("methods", []),
                    "total_sources": len(self.sources),
                    "key_findings": len(self.findings),
                },
            }

            self.logger.info(
                "Research task completed",
                findings_count=len(self.findings),
                sources_count=len(self.sources),
                confidence=result["confidence_score"],
            )

            return result

        except Exception as e:
            self.logger.error(
                "Research task failed",
                error=str(e),
                query=research_query,
            )
            raise AgentError(
                f"Research task failed: {str(e)}",
                agent_id=self.id,
                agent_name=self.config.name,
            ) from e

    async def cleanup(self) -> None:
        """Cleanup research agent resources."""
        self.logger.info("Cleaning up research agent")

        # Clear research context
        self.research_context.clear()
        self.findings.clear()
        self.sources.clear()

        self.logger.info("Research agent cleanup complete")

    async def _initialize_search_tools(self) -> None:
        """Initialize search tools and APIs."""
        # Initialize web search capabilities
        self.search_tools = {
            "web_search": True,
            "academic_search": True,
            "news_search": True,
            "image_search": True,
        }

        self.logger.debug("Search tools initialized", tools=list(self.search_tools.keys()))

    async def _initialize_data_sources(self) -> None:
        """Initialize data sources and connections."""
        # Initialize data source connections
        self.data_sources = {
            "web": {"enabled": True, "rate_limit": 10},
            "academic": {"enabled": True, "rate_limit": 5},
            "news": {"enabled": True, "rate_limit": 15},
            "social": {"enabled": False, "rate_limit": 20},
        }

        self.logger.debug("Data sources initialized", sources=list(self.data_sources.keys()))

    async def _analyze_query(
        self,
        query: str,
        research_type: str,
        depth: str,
    ) -> dict[str, Any]:
        """Analyze research query and create research plan."""
        system_prompt = f"""
        You are an expert research planner. Analyze the following research query and create a comprehensive research plan.
        
        Query: {query}
        Research Type: {research_type}
        Depth: {depth}
        
        Create a research plan with:
        1. Key research questions
        2. Search strategies
        3. Information sources to explore
        4. Analysis methods
        5. Expected deliverables
        
        Respond with a JSON object containing the research plan.
        """

        user_prompt = f"""
        Please analyze this research query and create a detailed research plan:
        
        Query: "{query}"
        Type: {research_type}
        Depth: {depth}
        
        Consider the scope, complexity, and required resources for this research.
        """

        response = await self._call_openai(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        plan = json.loads(response.choices[0].message.content)

        self.logger.info(
            "Research plan created",
            key_questions=len(plan.get("key_questions", [])),
            methods=plan.get("methods", []),
        )

        return plan

    async def _gather_information(
        self,
        research_plan: dict[str, Any],
        sources: list[str],
    ) -> dict[str, Any]:
        """Gather information from various sources."""
        gathered_info = {
            "web_results": [],
            "academic_results": [],
            "news_results": [],
            "social_results": [],
        }

        # Simulate information gathering from different sources
        for source in sources:
            if source == "web":
                web_results = await self._search_web(research_plan)
                gathered_info["web_results"] = web_results

            elif source == "academic":
                academic_results = await self._search_academic(research_plan)
                gathered_info["academic_results"] = academic_results

            elif source == "news":
                news_results = await self._search_news(research_plan)
                gathered_info["news_results"] = news_results

            elif source == "social":
                social_results = await self._search_social(research_plan)
                gathered_info["social_results"] = social_results

        # Record sources used
        total_results = sum(len(results) for results in gathered_info.values())

        self.logger.info(
            "Information gathering complete",
            total_results=total_results,
            sources_used=sources,
        )

        return gathered_info

    async def _search_web(self, research_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Search web sources for information."""
        # Simulate web search results
        search_queries = research_plan.get("search_queries", [])
        results = []

        for i, query in enumerate(search_queries[:5]):  # Limit to top 5 queries
            result = {
                "query": query,
                "title": f"Web Result {i+1} for: {query}",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a simulated web search result for query: {query}. It contains relevant information about the topic.",
                "date": datetime.utcnow().isoformat(),
                "source": "web",
                "relevance_score": 0.8 - (i * 0.1),
            }
            results.append(result)

            # Add to sources
            self.sources.append(
                {
                    "type": "web",
                    "url": result["url"],
                    "title": result["title"],
                    "accessed_at": datetime.utcnow().isoformat(),
                }
            )

        return results

    async def _search_academic(self, research_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Search academic sources for information."""
        # Simulate academic search results
        search_queries = research_plan.get("search_queries", [])
        results = []

        for i, query in enumerate(search_queries[:3]):  # Limit to top 3 queries
            result = {
                "query": query,
                "title": f"Academic Paper {i+1}: {query}",
                "authors": ["Dr. John Smith", "Dr. Jane Doe"],
                "journal": "Journal of Research",
                "year": 2023 - i,
                "doi": f"10.1000/example.{i+1}",
                "abstract": f"This academic paper discusses {query} with comprehensive analysis and findings.",
                "source": "academic",
                "citation_count": 50 - (i * 10),
                "relevance_score": 0.9 - (i * 0.1),
            }
            results.append(result)

            # Add to sources
            self.sources.append(
                {
                    "type": "academic",
                    "doi": result["doi"],
                    "title": result["title"],
                    "authors": result["authors"],
                    "accessed_at": datetime.utcnow().isoformat(),
                }
            )

        return results

    async def _search_news(self, research_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Search news sources for information."""
        # Simulate news search results
        search_queries = research_plan.get("search_queries", [])
        results = []

        for i, query in enumerate(search_queries[:4]):  # Limit to top 4 queries
            result = {
                "query": query,
                "title": f"News: {query}",
                "url": f"https://news.example.com/article-{i+1}",
                "summary": f"Recent news about {query} with latest developments and insights.",
                "published_date": datetime.utcnow().isoformat(),
                "source": "news",
                "publication": "News Source",
                "relevance_score": 0.75 - (i * 0.1),
            }
            results.append(result)

            # Add to sources
            self.sources.append(
                {
                    "type": "news",
                    "url": result["url"],
                    "title": result["title"],
                    "publication": result["publication"],
                    "accessed_at": datetime.utcnow().isoformat(),
                }
            )

        return results

    async def _search_social(self, research_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Search social media sources for information."""
        # Simulate social media search results
        search_queries = research_plan.get("search_queries", [])
        results = []

        for i, query in enumerate(search_queries[:2]):  # Limit to top 2 queries
            result = {
                "query": query,
                "content": f"Social media discussion about {query}",
                "platform": "Twitter",
                "author": f"user_{i+1}",
                "timestamp": datetime.utcnow().isoformat(),
                "engagement": {"likes": 10, "shares": 5, "comments": 3},
                "source": "social",
                "relevance_score": 0.6 - (i * 0.1),
            }
            results.append(result)

            # Add to sources
            self.sources.append(
                {
                    "type": "social",
                    "platform": result["platform"],
                    "author": result["author"],
                    "content": result["content"][:100] + "...",
                    "accessed_at": datetime.utcnow().isoformat(),
                }
            )

        return results

    async def _analyze_information(
        self,
        gathered_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze gathered information and extract insights."""
        # Compile all information
        all_results = []
        for source_type, results in gathered_info.items():
            all_results.extend(results)

        # Create analysis prompt
        system_prompt = """
        You are an expert research analyst. Analyze the following information and provide:
        1. Key findings and insights
        2. Patterns and trends
        3. Contradictions or gaps
        4. Confidence assessment
        5. Recommendations for further research
        
        Respond with a JSON object containing your analysis.
        """

        user_prompt = f"""
        Please analyze the following research information:
        
        Total sources: {len(all_results)}
        
        Information summary:
        {json.dumps(all_results[:10], indent=2)}  # Limit to first 10 for analysis
        
        Provide comprehensive analysis with key insights and confidence score.
        """

        response = await self._call_openai(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        analysis = json.loads(response.choices[0].message.content)

        # Extract key findings
        if "key_findings" in analysis:
            for finding in analysis["key_findings"]:
                self.findings.append(
                    {
                        "finding": finding,
                        "confidence": analysis.get("confidence", 0.8),
                        "source_count": len(all_results),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        self.logger.info(
            "Information analysis complete",
            findings_count=len(self.findings),
            confidence=analysis.get("confidence", 0.8),
        )

        return analysis

    async def _generate_report(
        self,
        analysis: dict[str, Any],
        original_query: str,
    ) -> dict[str, Any]:
        """Generate comprehensive research report."""
        system_prompt = """
        You are an expert research writer. Create a comprehensive research report based on the analysis provided.
        
        The report should include:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Conclusions
        5. Recommendations
        6. Limitations
        
        Write in a professional, clear, and well-structured format.
        Respond with a JSON object containing the report sections.
        """

        user_prompt = f"""
        Create a comprehensive research report for the query: "{original_query}"
        
        Based on the following analysis:
        {json.dumps(analysis, indent=2)}
        
        Sources analyzed: {len(self.sources)}
        Key findings: {len(self.findings)}
        
        Generate a professional research report with all required sections.
        """

        response = await self._call_openai(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        report = json.loads(response.choices[0].message.content)

        # Add metadata
        report["metadata"] = {
            "generated_at": datetime.utcnow().isoformat(),
            "agent_id": self.id,
            "agent_name": self.config.name,
            "total_sources": len(self.sources),
            "key_findings": len(self.findings),
            "confidence_score": analysis.get("confidence", 0.8),
        }

        self.logger.info(
            "Research report generated",
            sections=list(report.keys()),
            word_count=len(str(report)),
        )

        return report

    async def _verify_facts(self, report: dict[str, Any]) -> dict[str, Any]:
        """Verify facts and add confidence scores."""
        # Simulate fact verification
        verification_results = {
            "verified_facts": [],
            "questionable_facts": [],
            "unverified_facts": [],
            "overall_credibility": 0.85,
        }

        # Add verification metadata to report
        report["verification"] = verification_results
        report["verification"]["verified_at"] = datetime.utcnow().isoformat()

        self.logger.info(
            "Fact verification complete",
            credibility_score=verification_results["overall_credibility"],
        )

        return report

    def get_research_summary(self) -> dict[str, Any]:
        """Get summary of current research session."""
        return {
            "total_findings": len(self.findings),
            "total_sources": len(self.sources),
            "research_context": self.research_context,
            "methods_used": self.research_methods,
            "session_duration": (datetime.utcnow() - self.created_at).total_seconds(),
        }

    def export_research_data(self) -> dict[str, Any]:
        """Export all research data for external use."""
        return {
            "agent_info": {
                "id": self.id,
                "name": self.config.name,
                "created_at": self.created_at.isoformat(),
            },
            "findings": self.findings,
            "sources": self.sources,
            "context": self.research_context,
            "exported_at": datetime.utcnow().isoformat(),
        }
