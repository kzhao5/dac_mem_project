"""
DAC-Mem: Derivability-Aware Controller for Persistent Memory.

What Not to Remember: Tool-Conditioned Memory Writing
for Long-Horizon LLM Agents.
"""

from .schema import (
    ControllerState, EvaluationResult, Example, JudgeResult,
    MemoryItem, RetrievedItem, Turn,
)
from .controllers import (
    AMACLiteController, BaseController, ControllerConfig,
    DACMemController, NoveltyRecencyController,
    RelevanceOnlyController, StoreAllController,
    controller_from_name,
)
from .baselines import MemoryBankController, MemoryR1Controller
from .datasets import (
    load_dataset, load_locomo, load_longmemeval, load_perma,
    load_memorybench, load_synthetic, maybe_download_dataset,
)
from .embedding import BaseEmbedder, get_embedder
from .extraction import CandidateExtractor, LLMCandidateExtractor, ToolFactExtractor
from .evaluation import (
    LLMJudge, aggregate, bootstrap_ci, memory_quality_metrics,
    paired_bootstrap_test, qa_metrics, retrieval_metrics,
    significance_matrix,
)
from .features import FeatureComputer, SemanticFeatureComputer
from .llm import BaseLLM, get_llm
from .pipeline import MemoryPipeline, run_controller_on_examples
from .reader import LLMReader, make_reader
from .retrieval import HybridRetriever
from .tool_index import (
    LLMDerivabilityProbe, ToolGroundedRecoveryProbe,
    ToolIndex, relevant_tools_for_type,
)

__version__ = '0.2.0'
