# Роль файла: Общая схема состояния для графа.
from typing import TypedDict, Annotated
import operator


class Message(TypedDict):
    role: str   # "user" | "assistant"
    content: str


class Block(TypedDict):
    index: int
    topic: str
    goal: str
    subtopics: list[str]
    completed: bool


class State(TypedDict):
    user_id: int
    discipline: str
    level: str                              # beginner | intermediate | advanced | ""
    plan: list[Block]
    current_block_idx: int
    history: Annotated[list[Message], operator.add]
    xp: int
    achievements: Annotated[list[str], operator.add]
    mode: str                               # learning | quiz | qa | lesson_chat | idle | onboarding
    last_message: str                       # последнее сообщение пользователя
    response: str                           # ответ боту для отправки
    onboarding_step: int                    # индекс текущего вопроса онбординга
    onboarding_answers: list[str]           # валидные ответы онбординга по порядку
    last_block_content: str                 # контекст из RAG для текущего блока
    last_block_quiz_context: str            # текст именно показанного учебного блока для генерации квиза
    retry_count: int                        # сколько раз текущий блок объясняем повторно
    quiz_questions: list                    # вопросы текущего квиза
    quiz_current_idx: int                   # индекс текущего вопроса квиза
    quiz_answers: list[int]                 # выбранные ответы по порядку
    block_parts: list[str]                  # части текущего блока для степпера
    current_subtopic_idx: int               # индекс отображённой части блока
