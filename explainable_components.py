import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class DecisionExplanation:
    """决策解释数据类"""
    agent_name: str
    decision_type: str
    context: Dict[str, Any]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

class AgentInteractionLogger:
    """智能体交互记录器"""
    def __init__(self):
        self.interaction_history: List[Dict] = []
        
    def log_interaction(self, from_agent: str, to_agent: str, reason: str, context: Dict = None):
        """记录智能体之间的交互"""
        interaction = {
            "timestamp": datetime.now(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "context": context or {}
        }
        self.interaction_history.append(interaction)
        return interaction

    def get_interaction_history(self) -> List[Dict]:
        """获取交互历史"""
        return self.interaction_history

class LearningPathTracker:
    """学习路径追踪器"""
    def __init__(self):
        self.decision_points: List[Dict] = []
        
    def record_decision(self, agent_name: str, decision: str, rationale: str, impact: str):
        """记录学习路径上的关键决策点"""
        decision_point = {
            "timestamp": datetime.now(),
            "agent": agent_name,
            "decision": decision,
            "rationale": rationale,
            "impact": impact
        }
        self.decision_points.append(decision_point)
        return decision_point

    def get_learning_path(self) -> List[Dict]:
        """获取完整学习路径"""
        return self.decision_points

class FeedbackExplainer:
    """反馈解释器"""
    def generate_explanation(self, feedback_type: str, context: Dict) -> Dict:
        """生成结构化的反馈解释"""
        explanation = {
            "timestamp": datetime.now(),
            "type": feedback_type,
            "content": context.get("content", ""),
            "reasoning": context.get("reasoning", ""),
            "data_support": context.get("data_support", {}),
            "improvement_suggestions": context.get("suggestions", [])
        }
        return explanation 