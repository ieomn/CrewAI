import os
import json
from datetime import datetime
import time
import traceback  # 用于打印详细错误堆栈
from swarm import Swarm, Agent
from explainable_components import (
    AgentInteractionLogger,
    LearningPathTracker,
    FeedbackExplainer,
    DecisionExplanation
)
from repl.repl import run_demo_loop

# --- 配置 ---
# model = 'mistral'
model1 = 'gpt-4o-mini'
# model2 = 'gpt-4-turbo'
# model1 = 'deepseek-r1:8b'

course_name = "数据结构与算法"
rounds = 10
PROFILE_SAVE_DIR = "student_profiles"  # 档案保存目录

# 全局变量，用于持有当前活动学生的档案
current_student_profile = None

# --- StudentProfile 类 (增加序列化、ID、细化维度) ---
class StudentProfile:
    def __init__(self, student_id="default_student"): # 增加 student_id
        self.student_id = student_id # 学生ID
        # 基础属性初始化
        self.performance_history = []
        self.weak_points = set()
        self.strong_points = set()
        self.current_difficulty = "medium"
        self.total_questions = 0
        self.correct_answers = 0
        self.topic_scores = {}

        # 新增详细跟踪属性
        self.knowledge_points_mastery = {}
        self.question_type_performance = {}
        self.time_statistics = {
            "total_time": 0, "average_time": 0,
            "time_by_difficulty": {"easy": 0, "medium": 0, "hard": 0}
        }
        self.learning_behavior_stats = {
            "hint_usage": 0, "reference_usage": 0, "independent_solutions": 0
        }
        # --- 更新: 细化维度分数 ---
        self.dimension_scores = {
            # 核心能力维度
            "concept_understanding": [],
            "problem_solving": [],
            "code_implementation": [],
            # 态度/状态细化维度
            "effort_completeness": [], # 努力程度/完整性
            "engagement_participation": [], # 互动性/参与度
            "receptiveness_feedback": [], # 反馈接受度
            "communication_clarity": [], # 表达清晰度
            # 保留一个总体的学习态度，可以由 LLM 给出或后续计算得出 (0-10, 存储为0-1)
            "learning_attitude_overall": []
        }
        # --- 更新结束 ---

        # 可解释性相关属性初始化
        self.interaction_logger = AgentInteractionLogger()
        self.path_tracker = LearningPathTracker()
        self.feedback_explainer = FeedbackExplainer()
        self.decision_explanations = [] # DecisionExplanation 对象列表

    def _record_initial_state(self):
        """记录初始状态 (仅在新创建profile时调用)"""
        if not self.decision_explanations: # 避免重复记录
            initial_explanation = DecisionExplanation(
                agent_name="System", decision_type="initialization",
                context={
                    "student_id": self.student_id,
                    "difficulty": self.current_difficulty,
                    "total_questions": self.total_questions,
                    "correct_answers": self.correct_answers
                },
                reasoning=f"为学生 {self.student_id} 初始化档案，设置默认难度为medium"
            )
            self.decision_explanations.append(initial_explanation)
            self.path_tracker.record_decision(
                agent_name="System", decision="初始化学习状态",
                rationale=f"系统启动，为学生 {self.student_id} 设置初始参数",
                impact="建立初始学习环境，准备开始测试"
            )
            self.interaction_logger.log_interaction(
                from_agent="System", to_agent="Coordinator", reason="系统初始化",
                context={"student_id": self.student_id, "initial_difficulty": self.current_difficulty, "status": "ready"}
            )

    # --- 序列化方法 ---
    def to_dict(self):
        """将 StudentProfile 对象转换为可序列化字典"""
        serializable_explanations = []
        try:
            for exp in self.decision_explanations:
                serializable_explanations.append({
                    "agent_name": exp.agent_name,
                    "decision_type": exp.decision_type,
                    "context": exp.context, # 假设 context 本身是可序列化的
                    "reasoning": exp.reasoning,
                    "timestamp": exp.timestamp.isoformat() if isinstance(exp.timestamp, datetime) else str(exp.timestamp)
                })
        except Exception as e:
            print(f"[Warning] Error serializing decision explanations: {e}")

        return {
            "student_id": self.student_id,
            "performance_history": self.performance_history,
            "weak_points": list(self.weak_points),
            "strong_points": list(self.strong_points),
            "current_difficulty": self.current_difficulty,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "topic_scores": self.topic_scores,
            "knowledge_points_mastery": self.knowledge_points_mastery,
            "question_type_performance": self.question_type_performance,
            "time_statistics": self.time_statistics,
            "learning_behavior_stats": self.learning_behavior_stats,
            "dimension_scores": self.dimension_scores, # 包含新维度
            "decision_explanations": serializable_explanations,
            "interaction_log": [
                {**log, "timestamp": log["timestamp"].isoformat() if isinstance(log.get("timestamp"), datetime) else str(log.get("timestamp"))}
                for log in self.interaction_logger.get_interaction_history()
            ],
            "learning_path": [
                 {**step, "timestamp": step["timestamp"].isoformat() if isinstance(step.get("timestamp"), datetime) else str(step.get("timestamp"))}
                 for step in self.path_tracker.get_learning_path()
            ]
        }

    # --- 反序列化方法 ---
    @classmethod
    def from_dict(cls, data):
        """从字典创建 StudentProfile 对象"""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        student_id = data.get("student_id", "unknown_student")
        profile = cls(student_id)

        profile.performance_history = data.get("performance_history", [])
        profile.weak_points = set(data.get("weak_points", []))
        profile.strong_points = set(data.get("strong_points", []))
        profile.current_difficulty = data.get("current_difficulty", "medium")
        profile.total_questions = data.get("total_questions", 0)
        profile.correct_answers = data.get("correct_answers", 0)
        profile.topic_scores = data.get("topic_scores", {})
        profile.knowledge_points_mastery = data.get("knowledge_points_mastery", {})
        profile.question_type_performance = data.get("question_type_performance", {})
        profile.time_statistics = data.get("time_statistics", {
            "total_time": 0, "average_time": 0,
            "time_by_difficulty": {"easy": 0, "medium": 0, "hard": 0}
        })
        profile.learning_behavior_stats = data.get("learning_behavior_stats", {
            "hint_usage": 0, "reference_usage": 0, "independent_solutions": 0
        })
        # --- 更新: 恢复细化维度分数 ---
        # 使用预定义的维度作为模板，确保所有key都存在
        default_dims = {
            "concept_understanding": [], "problem_solving": [], "code_implementation": [],
            "effort_completeness": [], "engagement_participation": [],
            "receptiveness_feedback": [], "communication_clarity": [],
            "learning_attitude_overall": []
        }
        loaded_dims = data.get("dimension_scores", {})
        # 合并加载的数据和默认模板，加载的数据优先
        profile.dimension_scores = {**default_dims, **loaded_dims}
        # --- 更新结束 ---


        # 恢复可解释性组件的数据
        try:
            profile.decision_explanations = []
            for exp_data in data.get("decision_explanations", []):
                 ts = exp_data.get("timestamp")
                 timestamp_obj = None
                 if ts:
                     try:
                         timestamp_obj = datetime.fromisoformat(ts)
                     except ValueError:
                         print(f"[Warning] Could not parse timestamp '{ts}' for decision explanation. Using current time.")
                         timestamp_obj = datetime.now()
                 else:
                     timestamp_obj = datetime.now()

                 profile.decision_explanations.append(
                    DecisionExplanation(
                        agent_name=exp_data.get("agent_name"),
                        decision_type=exp_data.get("decision_type"),
                        context=exp_data.get("context"),
                        reasoning=exp_data.get("reasoning"),
                        timestamp=timestamp_obj
                    )
                 )

            interaction_log_data = data.get("interaction_log", [])
            for log_entry in interaction_log_data:
                ts = log_entry.get("timestamp")
                ts_override = None
                if ts:
                    try:
                        ts_override = datetime.fromisoformat(ts)
                    except ValueError:
                         print(f"[Warning] Could not parse timestamp '{ts}' for interaction log. Log time might be inaccurate.")
                profile.interaction_logger.log_interaction(
                    from_agent=log_entry.get("from_agent"), to_agent=log_entry.get("to_agent"),
                    reason=log_entry.get("reason"), context=log_entry.get("context"),
                    timestamp_override=ts_override
                )

            learning_path_data = data.get("learning_path", [])
            for step_entry in learning_path_data:
                 ts = step_entry.get("timestamp")
                 ts_override = None
                 if ts:
                     try:
                         ts_override = datetime.fromisoformat(ts)
                     except ValueError:
                          print(f"[Warning] Could not parse timestamp '{ts}' for learning path. Step time might be inaccurate.")
                 profile.path_tracker.record_decision(
                     agent_name=step_entry.get("agent_name"), decision=step_entry.get("decision"),
                     rationale=step_entry.get("rationale"), impact=step_entry.get("impact"),
                     timestamp_override=ts_override
                 )
        except Exception as e:
            print(f"[Warning] Error restoring explanation components from dict: {e}. Components might be reset.")
            profile.interaction_logger = AgentInteractionLogger()
            profile.path_tracker = LearningPathTracker()
            profile.decision_explanations = []

        return profile

    # --- 其他方法 ---
    def calculate_mastery(self):
        """计算知识掌握度"""
        if self.total_questions == 0:
            return 0.0
        mastery = self.correct_answers / self.total_questions

        # 减少重复记录，只在必要时记录
        # 例如，可以检查上一次记录的掌握度与本次是否有显著差异
        # 为简化，暂时保留每次计算都记录
        explanation = DecisionExplanation(
            agent_name="System", decision_type="mastery_calculation",
            context={
                "student_id": self.student_id,
                "total_questions": self.total_questions,
                "correct_answers": self.correct_answers,
                "mastery_score": mastery
            },
            reasoning=f"基于总题数{self.total_questions}和正确数{self.correct_answers}计算得到掌握度{mastery:.2%}"
        )
        # 避免解释列表过长，可以考虑限制长度或只保留最近的N条
        self.decision_explanations.append(explanation)
        return mastery

    def get_decision_history(self):
        """获取决策历史及解释"""
        return self.decision_explanations

    def update_knowledge_points_mastery(self, knowledge_points, score):
        """更新具体知识点的掌握度"""
        if not isinstance(knowledge_points, list):
            print(f"[Warning] knowledge_points received is not a list: {knowledge_points}. Skipping update.")
            return
        for point in knowledge_points:
            if not isinstance(point, str) or not point: # 增加空字符串检查
                 print(f"[Warning] Invalid or empty knowledge point: {point}. Skipping.")
                 continue
            point = point.strip() # 去除首尾空格
            if point not in self.knowledge_points_mastery:
                self.knowledge_points_mastery[point] = {"total_score": 0, "attempts": 0}
            try:
                numeric_score = float(score)
                self.knowledge_points_mastery[point]["total_score"] += numeric_score
                self.knowledge_points_mastery[point]["attempts"] += 1
            except (ValueError, TypeError):
                 print(f"[Warning] Invalid score type for knowledge point {point}: {score}. Skipping update.")

    def update_time_statistics(self, time_spent, difficulty):
        """更新时间统计"""
        try:
            time_spent_num = float(time_spent)
            if time_spent_num < 0: # 检查负时间
                 print(f"[Warning] Negative time_spent value received: {time_spent_num}. Using 0 instead.")
                 time_spent_num = 0

            self.time_statistics["total_time"] += time_spent_num
            self.time_statistics["average_time"] = (
                self.time_statistics["total_time"] / max(1, self.total_questions)
            )
            # 确保 difficulty 是字符串且有效
            difficulty_str = str(difficulty).lower().strip()
            if difficulty_str in self.time_statistics["time_by_difficulty"]:
                self.time_statistics["time_by_difficulty"][difficulty_str] += time_spent_num
            else:
                print(f"[Warning] Unknown or invalid difficulty level '{difficulty}' in time statistics. Ignoring.")
        except (ValueError, TypeError):
            print(f"[Warning] Invalid time_spent value: {time_spent}. Skipping time statistics update.")

    def update_learning_behavior(self, behavior_data):
        """更新学习行为统计"""
        if not isinstance(behavior_data, dict):
            print(f"[Warning] Invalid learning_behavior data format (not a dict): {behavior_data}. Skipping update.")
            return
        # 修正独立解题逻辑: used_hints 为 False 时才算独立
        if behavior_data.get("used_hints") is False:
            self.learning_behavior_stats["independent_solutions"] += 1
        if behavior_data.get("used_hints") is True: # 显式检查 True
            self.learning_behavior_stats["hint_usage"] += 1
        # 检查 reference_materials 是否是列表且非空
        refs = behavior_data.get("reference_materials")
        if isinstance(refs, list) and refs:
            self.learning_behavior_stats["reference_usage"] += 1
        elif refs is not None and not isinstance(refs, list):
             print(f"[Warning] reference_materials is not a list: {refs}. Not counting usage.")

    # --- 更新: update_dimension_scores 加入范围检查和归一化 ---
    def update_dimension_scores(self, assessment_dimensions):
        """更新各维度分数 (包含细化维度, 范围检查, 归一化)"""
        if not isinstance(assessment_dimensions, dict):
            print(f"[Warning] Invalid assessment_dimensions data format (not a dict): {assessment_dimensions}. Skipping update.")
            return

        for dimension, score in assessment_dimensions.items():
            if dimension in self.dimension_scores: # 检查key是否存在于预定义维度中
                try:
                    numeric_score = float(score)

                    # learning_attitude_overall 使用 0-10 评分, 存储时归一化到 0-1
                    if dimension == "learning_attitude_overall":
                        if 0 <= numeric_score <= 10:
                            self.dimension_scores[dimension].append(numeric_score / 10.0)
                        else:
                            print(f"[Warning] Score {numeric_score} for dimension {dimension} is out of expected range (0-10). Clamping.")
                            clamped_score = max(0.0, min(10.0, numeric_score))
                            self.dimension_scores[dimension].append(clamped_score / 10.0)
                    # 其他维度使用 0-1 评分
                    else:
                        if 0.0 <= numeric_score <= 1.0:
                            self.dimension_scores[dimension].append(numeric_score)
                        else:
                            print(f"[Warning] Score {numeric_score} for dimension {dimension} is out of expected range (0-1). Clamping.")
                            clamped_score = max(0.0, min(1.0, numeric_score))
                            self.dimension_scores[dimension].append(clamped_score)

                except (ValueError, TypeError):
                    print(f"[Warning] Invalid score type for dimension {dimension}: {score}. Skipping.")
            else:
                # 如果收到了未预定义的维度，可以选择记录下来或忽略
                print(f"[Warning] Received score for unknown assessment dimension: '{dimension}'. Ignoring.")
    # --- 更新结束 ---

# --- 文件操作函数 ---
def get_profile_path(student_id):
    """根据学生ID生成档案文件路径"""
    # 移除非法字符，保留字母、数字、下划线、短横线
    safe_id = "".join(c for c in str(student_id) if c.isalnum() or c in ['_', '-']).strip()
    if not safe_id: # 如果清理后为空
        safe_id = "invalid_id"
        print(f"[Warning] Provided student ID '{student_id}' resulted in an empty safe ID. Using 'invalid_id'.")
    elif safe_id != student_id:
         print(f"[Warning] Provided student ID '{student_id}' was sanitized to '{safe_id}'.")

    filename = f"student_{safe_id}.json"
    return os.path.join(PROFILE_SAVE_DIR, filename)

def load_or_create_profile(student_id):
    """加载学生档案，如果不存在或无效则创建新的"""
    profile_path = get_profile_path(student_id)
    print(f"正在尝试加载学生 '{student_id}' (路径: {profile_path}) 的档案...")

    try:
        os.makedirs(PROFILE_SAVE_DIR, exist_ok=True)
    except OSError as e:
        print(f"[Error] 无法创建档案目录 {PROFILE_SAVE_DIR}: {e}. 将在内存中运行。")
        profile = StudentProfile(student_id)
        profile._record_initial_state()
        return profile

    profile = None
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            profile = StudentProfile.from_dict(data)
            # 确保加载的 profile 的 student_id 与请求的一致
            if profile.student_id != student_id:
                 print(f"[Warning] Loaded profile ID '{profile.student_id}' does not match requested ID '{student_id}'. Using loaded ID.")
                 # 如果需要严格匹配，可以在这里抛出错误或创建新 profile
            print(f"成功加载学生 '{profile.student_id}' 的档案。")
            profile.interaction_logger.log_interaction("System", "System", "Profile Loaded", {"student_id": profile.student_id, "path": profile_path})
        except json.JSONDecodeError as e:
            print(f"[Error] 解析档案文件 {profile_path} 失败: {e}. 将创建新档案。")
            backup_path = f"{profile_path}.corrupted_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                os.rename(profile_path, backup_path)
                print(f"已将损坏的档案备份到: {backup_path}")
            except OSError as backup_e:
                print(f"[Error] 备份损坏的档案失败: {backup_e}")
        except (ValueError, TypeError, KeyError) as e: # 捕获 from_dict 可能抛出的错误
             print(f"[Error] 档案数据格式无效或缺失键: {e}. 将创建新档案。")
             print(traceback.format_exc()) # 打印堆栈帮助调试
             # 同样可以尝试备份
        except Exception as e:
            print(f"[Error] 加载档案时发生未知错误: {e}. 将创建新档案。")
            print(traceback.format_exc())

    if profile is None:
        print(f"为学生 '{student_id}' 创建新的档案。")
        profile = StudentProfile(student_id)
        profile._record_initial_state()
        save_profile(profile) # 尝试保存新创建的档案

    return profile

def save_profile(profile):
    """将学生档案保存到JSON文件"""
    if not isinstance(profile, StudentProfile):
        print("[Error] 提供给 save_profile 的不是有效的 StudentProfile 对象。")
        return False

    profile_path = get_profile_path(profile.student_id)
    print(f"正在保存学生 '{profile.student_id}' 的档案到: {profile_path} ...")
    try:
        profile_dict = profile.to_dict()
        temp_path = profile_path + ".tmp" # 使用临时文件保证原子性

        # 确保目录存在
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, ensure_ascii=False, indent=4)

        # 替换旧文件
        os.replace(temp_path, profile_path) # 原子操作 (在大多数系统上)
        print(f"档案保存成功。")
        return True
    except IOError as e:
        print(f"[Error] 无法写入档案文件 {profile_path} (或临时文件): {e}")
    except TypeError as e:
        print(f"[Error] 序列化档案时出错 (可能存在非JSON兼容类型): {e}")
        # 尝试打印出问题的部分数据 (如果可能)
        # for k, v in profile_dict.items():
        #     try: json.dumps(v)
        #     except TypeError: print(f"Error serializing key: {k}")
    except Exception as e:
        print(f"[Error] 保存档案时发生未知错误: {e}")
        print(traceback.format_exc())
    # 清理临时文件 (如果存在且替换失败)
    if os.path.exists(temp_path):
        try: os.remove(temp_path)
        except OSError: pass
    return False

# --- 核心更新函数 (使用全局档案, 触发保存, 更新难度逻辑) ---
def update_student_profile(performance_data):
    """更新当前活动学生的档案，并保存到文件"""
    global current_student_profile
    if current_student_profile is None:
        return {"status": "error", "message": "当前没有活动的學生档案。", "context": "更新学生档案操作失败"}

    try:
        # 数据验证
        required_keys = [
            "correct", "topic", "score", "question_details",
            "answer_process", "learning_behavior", "assessment_dimensions"
        ]
        if not isinstance(performance_data, dict):
             raise TypeError("传入的 performance_data 不是一个字典。")
        for key in required_keys:
            if key not in performance_data:
                missing_info = f"缺少必需的键: {key}。 传入的数据: {performance_data}"
                raise KeyError(missing_info)

        # --- 更新基础数据 ---
        current_student_profile.performance_history.append(performance_data)
        current_student_profile.total_questions += 1
        topic = performance_data.get("topic", "未知主题")

        q_details = performance_data.get("question_details", {})
        if not isinstance(q_details, dict):
             print(f"[Warning] question_details 格式不正确 (不是字典): {q_details}. 部分更新可能失败。")
             q_details = {}
        knowledge_points = q_details.get("knowledge_points", [])
        if not isinstance(knowledge_points, list):
             print(f"[Warning] knowledge_points 不是列表: {knowledge_points}. 将其视为空列表。")
             knowledge_points = []
        score = 0
        try:
            score = float(performance_data.get("score", 0))
        except (ValueError, TypeError):
             print(f"[Warning] score 值无效: {performance_data.get('score')}. 将使用 0。")

        current_student_profile.update_knowledge_points_mastery(knowledge_points, score)

        time_spent = q_details.get("time_spent", 0)
        difficulty = q_details.get("difficulty_level", "unknown")
        current_student_profile.update_time_statistics(time_spent, difficulty)

        learning_behavior = performance_data.get("learning_behavior", {})
        if not isinstance(learning_behavior, dict):
            print(f"[Warning] learning_behavior 不是字典: {learning_behavior}. 更新将跳过。")
            learning_behavior = {}
        current_student_profile.update_learning_behavior(learning_behavior)

        # --- 更新维度分数 (调用已修改的 StudentProfile 方法) ---
        assessment_dimensions = performance_data.get("assessment_dimensions", {})
        if not isinstance(assessment_dimensions, dict):
             print(f"[Warning] assessment_dimensions 不是字典: {assessment_dimensions}. 更新将跳过。")
             assessment_dimensions = {}
        # update_dimension_scores 会处理新维度和范围检查
        current_student_profile.update_dimension_scores(assessment_dimensions)

        # --- 记录答题决策 ---
        answer_explanation = DecisionExplanation(
            agent_name="System", decision_type="answer_recording",
            context={
                "student_id": current_student_profile.student_id,
                "topic": topic, "correct": performance_data.get("correct"),
                "score": score, "question_details": q_details,
                "answer_process": performance_data.get("answer_process", {}),
                "assessment_dimensions": assessment_dimensions # 记录本次评估的维度
            },
            reasoning=f"记录学生在{topic}主题上的详细答题表现，涉及知识点：{', '.join(knowledge_points)}"
        )
        current_student_profile.decision_explanations.append(answer_explanation)

        # 更新正确答案计数
        if performance_data.get("correct") is True:
            current_student_profile.correct_answers += 1

        # 更新知识点得分
        current_topic_score = current_student_profile.topic_scores.get(topic, 0)
        try: current_score_num = float(current_topic_score)
        except (ValueError, TypeError): current_score_num = 0
        new_score = current_score_num + score
        current_student_profile.topic_scores[topic] = new_score

        # --- 动态调整难度 (基于核心能力维度) ---
        mastery = current_student_profile.calculate_mastery()
        old_difficulty = current_student_profile.current_difficulty

        performance_related_dimensions = [
            "concept_understanding", "problem_solving", "code_implementation"
        ]
        valid_performance_scores = []
        # 从本次评估数据 assessment_dimensions 中获取核心能力分数
        for dim_key in performance_related_dimensions:
            dim_score = assessment_dimensions.get(dim_key)
            if isinstance(dim_score, (int, float)):
                # 确保在0-1之间 (因为update_dimension_scores已处理范围)
                valid_performance_scores.append(max(0.0, min(1.0, float(dim_score))))

        avg_performance_score = 0
        if valid_performance_scores:
            try: avg_performance_score = sum(valid_performance_scores) / len(valid_performance_scores)
            except ZeroDivisionError: avg_performance_score = 0
        else: print("[Warning] 本次评估未提供有效的核心能力维度分数，难度调整可能不准确。")

        # 难度调整逻辑
        if mastery > 0.8 and avg_performance_score > 0.8: new_difficulty = "hard"
        elif mastery < 0.4 or avg_performance_score < 0.4: new_difficulty = "easy"
        else: new_difficulty = "medium"

        if old_difficulty != new_difficulty:
            current_student_profile.current_difficulty = new_difficulty
            difficulty_explanation = DecisionExplanation(
                agent_name="System", decision_type="difficulty_adjustment",
                context={
                    "student_id": current_student_profile.student_id,
                    "old_difficulty": old_difficulty, "new_difficulty": new_difficulty,
                    "mastery": mastery,
                    "avg_performance_score_used": avg_performance_score,
                    "performance_details": performance_data
                },
                reasoning=f"基于掌握度({mastery:.2%})和核心能力平均分({avg_performance_score:.2%})调整难度为{new_difficulty}"
            )
            current_student_profile.decision_explanations.append(difficulty_explanation)
        # --- 难度调整结束 ---

        # --- 触发保存 ---
        if not save_profile(current_student_profile):
             print("[Warning] 更新学生档案后保存到文件失败。")

        # --- 返回详细的更新结果 ---
        historical_averages = {}
        for dim, scores in current_student_profile.dimension_scores.items():
             if scores:
                 avg = sum(scores) / len(scores)
                 # 对 overall attitude 特殊处理 (存储的是0-1, 显示时乘以10)
                 if dim == "learning_attitude_overall":
                     historical_averages[dim] = f"{avg * 10:.1f}/10"
                 else:
                     historical_averages[dim] = f"{avg:.2f}/1.0"
             else: historical_averages[dim] = "N/A"

        explanation_output = {
            "mastery": mastery,
            "difficulty_change": {
                "old": old_difficulty, "new": current_student_profile.current_difficulty,
                "reason": f"基于掌握度({mastery:.2%})和核心能力平均分({avg_performance_score:.2%})的调整"
            },
            "topic_progress": {
                "topic": topic, "knowledge_points": knowledge_points,
                "old_score": current_score_num, "new_score": new_score
            },
            "detailed_assessment": assessment_dimensions, # 本次评估的详细维度
            "learning_behavior": current_student_profile.learning_behavior_stats,
            "time_statistics": current_student_profile.time_statistics,
            "historical_dimension_averages": historical_averages # 各维度历史平均分
        }
        return {"status": "updated", "current_difficulty": current_student_profile.current_difficulty, "explanation": explanation_output}

    except KeyError as e:
        print(f"[Error] 更新学生档案时缺少必需的数据: {e}")
        return {"status": "error", "message": f"缺少必需的数据: {e}", "context": "数据验证失败"}
    except TypeError as e:
         print(f"[Error] 更新学生档案时遇到类型错误: {e}")
         return {"status": "error", "message": f"数据类型错误: {e}", "context": "数据处理失败"}
    except Exception as e:
        print(f"[Critical Error] 更新学生档案时发生未预料的错误: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": f"发生内部错误: {e}", "context": "更新学生档案时发生严重错误"}

# --- 获取摘要函数 (使用全局档案, 展示细化维度) ---
def get_student_profile_summary():
    """获取当前活动学生档案摘要"""
    global current_student_profile
    if current_student_profile is None: return "错误：当前没有活动的學生档案。"

    try:
        mastery = current_student_profile.calculate_mastery()
        learning_path = current_student_profile.path_tracker.get_learning_path()
        interactions = current_student_profile.interaction_logger.get_interaction_history()
        decision_history = current_student_profile.get_decision_history()

        def format_timestamp(dt_object):
            if isinstance(dt_object, datetime): return dt_object.strftime("%Y-%m-%d %H:%M:%S")
            return str(dt_object)

        # 计算各维度历史平均分
        historical_averages = {}
        for dim, scores in current_student_profile.dimension_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                if dim == "learning_attitude_overall": historical_averages[dim] = f"{avg * 10:.1f}/10"
                else: historical_averages[dim] = f"{avg:.2f}/1.0"
            else: historical_averages[dim] = "N/A"

        summary = {
            "学生ID": current_student_profile.student_id,
            "基本信息": {
                "总答题数": current_student_profile.total_questions, "正确率": f"{mastery:.2%}",
                "当前难度": current_student_profile.current_difficulty,
            },
            "知识掌握": {
                "知识点得分": current_student_profile.topic_scores,
                "强项": list(current_student_profile.strong_points) or [],
                "弱项": list(current_student_profile.weak_points) or []
            },
            "详细维度平均表现": historical_averages, # 加入细化维度平均分
            "学习轨迹": {
                "学习路径决策点": [ # 来自 path_tracker
                    {"时间": format_timestamp(item.get("timestamp")), "决策": item.get("decision"),
                     "原因": item.get("rationale"), "影响": item.get("impact")}
                    for item in (learning_path or []) # 保证是列表
                ],
                "智能体交互": [
                    {"时间": format_timestamp(item.get("timestamp")), "从": item.get("from_agent"),
                     "到": item.get("to_agent"), "原因": item.get("reason")}
                    for item in (interactions[-5:] if interactions else []) # 最近5次
                ],
                "系统决策解释": [ # 来自 decision_explanations
                    {"时间": format_timestamp(exp.timestamp), "决策类型": exp.decision_type, "解释": exp.reasoning}
                    for exp in (decision_history[-5:] if decision_history else []) # 最近5个
                ]
            }
        }

        # 格式化输出
        output = f"--- 学生档案摘要 ({summary['学生ID']}) ---\n\n"
        output += f"基本信息：\n"
        output += f"- 总答题数：{summary['基本信息']['总答题数']}\n- 正确率：{summary['基本信息']['正确率']}\n- 当前难度：{summary['基本信息']['当前难度']}\n\n"
        output += f"知识掌握情况：\n"
        output += f"- 知识点得分：{json.dumps(summary['知识掌握']['知识点得分'], indent=2, ensure_ascii=False)}\n"
        output += f"- 强项：{summary['知识掌握']['强项'] or '暂无'}\n- 弱项：{summary['知识掌握']['弱项'] or '暂无'}\n\n"
        output += f"详细维度平均表现：\n"
        for dim, avg_score in summary['详细维度平均表现'].items():
             output += f"- {dim.replace('_', ' ').capitalize()}: {avg_score}\n" # 格式化维度名称
        output += "\n"
        output += f"最近的学习轨迹：\n"
        output += f"1. 学习路径决策点：\n{json.dumps(summary['学习轨迹']['学习路径决策点'], indent=2, ensure_ascii=False)}\n\n"
        output += f"2. 最近的智能体交互：\n{json.dumps(summary['学习轨迹']['智能体交互'], indent=2, ensure_ascii=False)}\n\n"
        output += f"3. 最近的系统决策解释：\n{json.dumps(summary['学习轨迹']['系统决策解释'], indent=2, ensure_ascii=False)}\n"
        output += "--- 摘要结束 ---\n"
        return output

    except Exception as e:
        print(f"[Critical Error] 获取学生档案摘要时发生错误: {e}")
        print(traceback.format_exc())
        return f"获取学生档案摘要时出错：{str(e)}"

# --- 状态转移函数 (使用全局档案) ---
# (这些函数基本不变，除了内部使用 current_student_profile 和增加错误处理)
def transfer_to_coordinator():
    global current_student_profile
    if current_student_profile is None: return coordinator
    time.sleep(1)
    try:
        current_student_profile.interaction_logger.log_interaction(
            from_agent=course_tutor.name, to_agent=coordinator.name,
            reason="完成当前轮次辅导，返回协调员进行下一步安排",
            context={
                "student_id": current_student_profile.student_id,
                "current_difficulty": current_student_profile.current_difficulty,
                "current_mastery": current_student_profile.calculate_mastery() # 重新计算确保最新
            }
        )
        current_student_profile.path_tracker.record_decision(
            agent_name=course_tutor.name, decision="转换到协调员",
            rationale="当前轮次辅导完成", impact="协调员将决定是继续下一轮还是结束测试"
        )
    except Exception as e: print(f"[Error] Logging transfer to coordinator failed: {e}")
    return coordinator

def transfer_to_tester():
    global current_student_profile
    if current_student_profile is None: return course_tester
    time.sleep(1)
    try:
        current_student_profile.interaction_logger.log_interaction(
            from_agent=coordinator.name, to_agent=course_tester.name,
            reason="开始新一轮测试",
            context={
                "student_id": current_student_profile.student_id,
                "current_difficulty": current_student_profile.current_difficulty,
                "strong_points": list(current_student_profile.strong_points),
                "weak_points": list(current_student_profile.weak_points)
            }
        )
        current_student_profile.path_tracker.record_decision(
            agent_name=coordinator.name, decision="转换到测试官",
            rationale="需要进行新一轮的知识测试", impact="测试官将根据当前难度和知识点情况出题"
        )
    except Exception as e: print(f"[Error] Logging transfer to tester failed: {e}")
    print("接下来，我出题考核。")
    time.sleep(1.5)
    return course_tester

def transfer_to_tutor():
    global current_student_profile
    if current_student_profile is None: return course_tutor
    time.sleep(1)
    try:
        last_perf = current_student_profile.performance_history[-1] if current_student_profile.performance_history else None
        context_perf_summary = None
        if last_perf and isinstance(last_perf, dict): # 确保 last_perf 是字典
            context_perf_summary = {
                "topic": last_perf.get("topic"), "correct": last_perf.get("correct"),
                "score": last_perf.get("score")
             }

        current_student_profile.interaction_logger.log_interaction(
            from_agent=course_tester.name, to_agent=course_tutor.name,
            reason="完成测试，需要辅导",
            context={
                "student_id": current_student_profile.student_id,
                "last_performance_summary": context_perf_summary,
                "current_difficulty": current_student_profile.current_difficulty
            }
        )
        current_student_profile.path_tracker.record_decision(
            agent_name=course_tester.name, decision="转换到辅导员",
            rationale="学生完成测试，需要针对性辅导", impact="辅导员将根据测试表现提供相应的指导"
        )
    except Exception as e: print(f"[Error] Logging transfer to tutor failed: {e}")
    print("接下来，我将把你转给课程辅导员进行进一步的辅导。")
    time.sleep(1.5)
    return course_tutor

def transfer_to_grader():
    global current_student_profile
    if current_student_profile is None: return final_grader
    time.sleep(1)
    try:
        current_student_profile.interaction_logger.log_interaction(
            from_agent=coordinator.name, to_agent=final_grader.name,
            reason="完成所有测试轮次，进行最终评估",
            context={
                "student_id": current_student_profile.student_id, "total_rounds": rounds,
                "final_difficulty": current_student_profile.current_difficulty,
                "mastery": current_student_profile.calculate_mastery(), # 重新计算
                "topic_scores": current_student_profile.topic_scores
            }
        )
        current_student_profile.path_tracker.record_decision(
            agent_name=coordinator.name, decision="转换到最终评分官",
            rationale=f"已完成{rounds}轮测试，需要进行综合评估", impact="最终评分官将对整体学习表现进行多维度评估"
        )
    except Exception as e: print(f"[Error] Logging transfer to grader failed: {e}")
    print("正在转到最终评分官...")
    time.sleep(1.5)
    return final_grader

def transfer_to_exit():
    global current_student_profile
    if current_student_profile is None:
        print("测试结束，退出（无活动档案）")
        exit()

    student_id_on_exit = current_student_profile.student_id # 先保存ID
    print(f"正在为学生 {student_id_on_exit} 结束测试流程...")
    try:
        print("正在进行最后一次档案保存...")
        save_success = save_profile(current_student_profile) # 保存包含所有最终状态的档案

        # 记录退出日志 (即使保存失败也要尝试记录)
        final_mastery = current_student_profile.calculate_mastery()
        current_student_profile.interaction_logger.log_interaction(
            from_agent=final_grader.name, to_agent="System",
            reason="完成所有评估，结束测试",
            context={
                "student_id": student_id_on_exit,
                "final_state": {
                    "total_questions": current_student_profile.total_questions, "mastery": final_mastery,
                    "final_difficulty": current_student_profile.current_difficulty,
                    "strong_points": list(current_student_profile.strong_points),
                    "weak_points": list(current_student_profile.weak_points)
                },
                "save_status": "Success" if save_success else "Failed"
            }
        )
        current_student_profile.path_tracker.record_decision(
            agent_name=final_grader.name, decision="结束测试",
            rationale="已完成所有测试和评估环节", impact="生成最终评估报告并结束测试流程"
        )
        # 尝试再次保存以包含退出日志 (如果第一次失败，这次可能也失败)
        if not save_success:
            print("尝试再次保存以记录退出日志...")
            save_profile(current_student_profile)

    except Exception as e:
        print(f"[Error] Logging or saving during exit failed: {e}")
        print(traceback.format_exc())

    print(f"学生 {student_id_on_exit} 的测试已结束。退出程序。")
    exit()

# --- 性能数据模板 (更新: 加入细化态度维度) ---
PERFORMANCE_DATA_TEMPLATE = '''{{
    "correct": false,
    "topic": "知识点名称",
    "score": 0, // 内容得分 (0-10)
    "question_details": {{ "question_type": "题目类型", "difficulty_level": "当前难度级别", "knowledge_points": ["知识点1", "知识点2"], "time_spent": 300 }},
    "answer_process": {{ "attempts": 1, "error_types": ["错误类型"], "code_quality": {{ "time_complexity": "N/A", "space_complexity": "N/A", "code_style": "N/A" }} }},
    "learning_behavior": {{ "used_hints": false, "reference_materials": [], "solution_approach": "解题方式" }},
    "assessment_dimensions": {{
        "concept_understanding": 0.6, // 概念理解 (0-1)
        "problem_solving": 0.5,       // 问题解决 (0-1)
        "code_implementation": 0.0,   // 代码实现 (0-1, 如果适用)
        "effort_completeness": 0.7,   // 努力/完整性 (0-1) - 回答是否详尽，是否尝试解释
        "engagement_participation": 0.8, // 参与度 (0-1) - 是否回应反馈，是否提问
        "receptiveness_feedback": 0.9, // 反馈接受度 (0-1) - 对纠错/建议的反应
        "communication_clarity": 0.7,  // 表达清晰度 (0-1)
        "learning_attitude_overall": 7.5 // 综合学习态度 (0-10) - 可选，作为整体印象分
    }}
}}'''

# --- 示例调用 (更新) ---
EXAMPLE_CALL = '''
update_student_profile({{
    "correct": False, "topic": "链表", "score": 3, // 内容得分较低
    "question_details": {{ "question_type": "概念题", "difficulty_level": "medium", "knowledge_points": ["链表", "时间复杂度"], "time_spent": 300 }},
    "answer_process": {{ "attempts": 1, "error_types": ["概念理解错误"], "code_quality": {{}} }},
    "learning_behavior": {{ "used_hints": false, "reference_materials": [], "solution_approach": "独立思考但方向错误" }},
    "assessment_dimensions": {{
        "concept_understanding": 0.4,
        "problem_solving": 0.5,
        "code_implementation": 0.0,
        "effort_completeness": 0.8,   // 虽然错了，但回答很详细，尝试解释了
        "engagement_participation": 0.9, // 在辅导员解释后，问了“为什么”
        "receptiveness_feedback": 0.9, // 接受了辅导员的纠正，并表示感谢
        "communication_clarity": 0.6,  // 解释有点绕，但能明白意思
        "learning_attitude_overall": 8 // 综合看态度积极 (0-10分)
    }}
}})
'''

# --- 智能体定义 (更新 instructions) ---

# 课程协调员 (不变)
coordinator = Agent(
    name="协调员", model=model1,
    instructions=f"""你是一位{course_name}课程的协调员，负责主持和推进整个课程测试流程。
在开始测试之前，你必须按照以下步骤操作：
1. 首先向学生介绍自己："你好，我是{course_name}课程的协调员。"
2. 然后介绍测试流程："本次测试共{rounds}轮，每轮包含以下环节：课程测试官出题考核、课程辅导员进行辅导。最后由最终评分官进行综合评估。"
3. 确认学生是否准备好开始测试。
4. 使用 transfer_to_tester() 将学生转给课程测试官。
5. 在每轮测试结束后，检查是否完成所有轮数（你可以通过与学生的对话轮数来大致判断，但主要依赖系统逻辑控制）。当前轮数信息可能不直接可见，按流程在 tutor 返回后判断是否需要转 grader。假设系统会在恰当时机让你判断。如果你认为轮数已到（比如经过了多次 tutor <-> tester 循环），并且 tutor 返回了，则执行：
   a. 说："现在{rounds}轮测试已全部完成，我将把你转到最终评分官进行综合评估。"
   b. 使用 transfer_to_grader() 转至最终评分官
   否则（如果 tutor 返回且你认为轮数未到）：
   c. 使用 transfer_to_tester() 继续下一轮
记住：要清晰地介绍每个环节的目的，严格控制轮数（理论上由外部循环控制，你的决策点是 tutor 返回后），{rounds}轮结束后必须转到最终评分官，不要说无关的话，必须调用 transfer_to_tester() 或 transfer_to_grader()。
""",
    functions=[transfer_to_tester, transfer_to_grader]
)

# 课程测试官 (不变)
course_tester = Agent(
    name = "测试官", model = model1,
    instructions = f"""你是一位{course_name}课程的测试官，负责考察学生对课程知识的掌握程度。
在开始出题之前，你必须按照以下步骤操作：
1. 首先说："让我先查看学生的历史表现记录。"
2. 然后调用 get_student_profile_summary() 获取学生档案，并仔细阅读结果（特别是当前难度、弱项）。
3. 根据获得的信息，说："基于以上分析，我来出一道适合的题目。"
4. 然后根据学生档案和分析结果出一道适合的题目（类型：选择、填空、简答、程序纠错、判断）。题目难度应匹配档案中的'当前难度'，优先考察'弱项'知识点。
5. 等学生结束回答后，必须使用 transfer_to_tutor() 将学生转给课程辅导员。
记住：每次只能出一道题，不要说无关的话，难度根据档案调整，优先考察弱项，避免重复强项，必须使用 transfer_to_tutor()。
""",
    functions=[get_student_profile_summary, update_student_profile, transfer_to_tutor] # 注意：tester 不应该调用 update_student_profile，评估由 tutor 做。这里保留可能是为了某种特殊情况？但正常流程不调用。
)

# --- 更新: 课程辅导员 instructions ---
course_tutor = Agent(
    name="辅导员",
    model=model1,
    instructions=f"""你是一位{course_name}课程的辅导员，负责在课程测试结束后，为学生提供总结性的评估和指导。

在开始辅导之前，你必须按照以下步骤操作：
1. 首先说："让我查看一下你刚才的答题表现。"
2. 调用 get_student_profile_summary() 获取学生档案。
3. 说："让我分析一下你的表现。"
4. 根据学生的回答和档案分析结果，进行针对性辅导：讲解错误，补充知识点，提供建议。**在辅导过程中，密切观察学生的互动方式和回应。**
5. 等待学生回应，确认是否理解（例如学生说"没有了"或表示理解）。

6. 辅导后，使用 update_student_profile(performance_data) 更新学生表现数据。
   **在构造 performance_data 时，除了评估内容本身 (correct: true/false, score: 0-10)，你还需要根据学生在本轮的互动表现，对以下维度进行评分 (0.0 - 1.0，learning_attitude_overall 除外是 0-10)：**
   - `concept_understanding`: 对概念的理解程度 (0-1)。
   - `problem_solving`: 解决问题的能力 (0-1)。
   - `code_implementation`: 代码实现能力 (0-1, 如果题目涉及)。
   - **`effort_completeness`**: 学生回答的努力程度和完整性 (0.0: 非常敷衍, 1.0: 非常详尽)。
   - **`engagement_participation`**: 学生在互动中的参与度 (0.0: 完全被动, 1.0: 非常积极)。
   - **`receptiveness_feedback`**: 学生对你纠错或建议的接受程度 (0.0: 抵触, 1.0: 欣然接受)。
   - **`communication_clarity`**: 学生表达的清晰度 (0.0: 难以理解, 1.0: 非常清晰)。
   - **`learning_attitude_overall`**: 你对学生本轮综合学习态度的整体印象分 (0-10)。

   **评分要客观，基于学生本轮的实际言行。**

   数据格式示例:
{PERFORMANCE_DATA_TEMPLATE}

   调用示例:
{EXAMPLE_CALL}

7. 在确认学生理解并完成辅导后，使用 transfer_to_coordinator() 将学生转回课程协调员。

记住：准确评估内容和态度，评分客观，耐心讲解，不要说无关的话，记录反馈，确保理解，必须等学生回应后再转换，必须使用 transfer_to_coordinator()，不要说"我将把你转回协调员"，直接执行转换，update_student_profile() 参数必须是完整的字典对象。
内容评分标准：correct=true (完全正确/基本正确/理解核心概念)；correct=false (完全错误/重大概念错误/未理解要点)。score (0-10) 反映内容质量。
态度维度评分标准：根据上述描述，给出 0.0 到 1.0 (或 0-10 for overall) 的分数。
""",
    functions=[
        get_student_profile_summary,
        update_student_profile,
        transfer_to_coordinator
    ]
)
# --- 更新结束 ---

# --- 更新: 最终评分官 instructions ---
final_grader = Agent(
    name="最终评分官",
    model=model1,
    instructions=f"""你是一位{course_name}课程的最终评分官，负责对学生的整体表现进行综合评估。

在开始评估之前，你必须严格按照以下步骤操作：
1. 首先说："让我查看一下你的整体测试记录。"
2. 调用 get_student_profile_summary() 获取学生完整档案和各项平均表现。**请特别注意'详细维度平均表现'部分，它总结了学生在整个测试中的平均水平。**
3. 说："让我对你的表现进行综合评估。"
4. 必须详细分析学生档案中的每项指标，包括：总答题数、正确率、最终难度、知识点得分、强项、弱项，以及各详细维度的平均表现。

5. 然后必须按以下维度给出具体的评分(A-E五级)和详细说明：
   A. 知识掌握程度 (基于 concept_understanding, problem_solving 的平均分和 topic_scores)
   B. 问题解决能力 (基于 problem_solving 平均分和具体题目表现历史 - 可参考 summary 中的信息)
   C. 代码实现能力 (基于 code_implementation 平均分和题目历史 - 如果适用)
   **D. 学习态度 (A-E五级)：**
      - **综合考虑学生档案中 'effort_completeness', 'engagement_participation', 'receptiveness_feedback', 'communication_clarity' 的历史平均分以及 'learning_attitude_overall' 的平均分（显示为 /10）。**
      - 评估学生在整个测试过程中的努力程度、参与度、对反馈的态度以及沟通表达情况。
      - 给出具体的例子或趋势（如果能从历史记录推断）来说明你的评分依据。例如：“从记录看，你在接受反馈方面一直表现积极（receptiveness_feedback 平均分高），但在问题阐述清晰度（communication_clarity 平均分低）上还有提升空间。”
   E. 总体评级 (A-E五级)：综合以上四个维度给出总体评价。

6. 必须提供详细的评价报告，包括：在各维度上的具体表现和分数/评级，明确指出优势领域和具体表现，指出需要改进的具体方面，给出针对性的学习建议（特别是针对薄弱环节和可改进的态度方面）。

7. 最后总结："以上就是我的评估报告，你有什么问题吗？"

8. 等待学生回应，如果学生表示没有问题或理解了评估内容，才能使用 transfer_to_exit() 结束测试流程。

评分标准：A(优秀, 对应平均分约0.9-1.0或9-10), B(良好, 约0.8-0.89或8-8.9), C(中等, 约0.7-0.79或7-7.9), D(及格, 约0.6-0.69或6-6.9), E(不及格, <0.6或<6)。请根据平均分进行合理映射。

注意事项：评价客观公正基于档案事实，各维度说明详细，建议具体可行，完成评估并等待回应后才能退出。
""",
    functions=[get_student_profile_summary, transfer_to_exit]
)
# --- 更新结束 ---

# --- 主程序入口 ---
if __name__ == "__main__":
    print("--- AI 智能教学系统 ---")
    student_id = ""
    while not student_id:
        student_id_input = input("请输入您的学生ID (例如: A, B, student01): ").strip()
        # 基本的ID有效性检查 (不允许空)
        if student_id_input:
            student_id = student_id_input
        else:
            print("学生ID不能为空，请重新输入。")

    try:
        current_student_profile = load_or_create_profile(student_id)
        if current_student_profile is None:
             print(f"[Fatal Error] 无法加载或创建学生 {student_id} 的档案。程序无法继续。")
             exit(1)
        print(f"已成功设置学生 '{current_student_profile.student_id}' 的档案为当前活动档案。") # 使用 profile 中的 ID
    except Exception as e:
        print(f"[Fatal Error] 初始化学生档案时发生严重错误: {e}")
        print(traceback.format_exc())
        exit(1)

    print("\n--- 测试即将开始 ---")
    try:
        run_demo_loop(coordinator, stream=False, debug=False)
    except KeyboardInterrupt:
         print("\n用户中断了程序。正在尝试保存当前进度...")
         if current_student_profile:
             save_profile(current_student_profile)
         print("程序已退出。")
    except Exception as e:
         print(f"\n[Fatal Error] 程序运行中发生未处理的错误: {e}")
         print(traceback.format_exc())
         print("正在尝试保存当前进度...")
         if current_student_profile:
             save_profile(current_student_profile)
         print("程序异常终止。")
         exit(1)