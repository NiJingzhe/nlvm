from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str
    title: str
    details: str | None = None
    effort: int = Field(..., ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    due_date: date | None = None
    done: bool = False
    created_at: datetime
    updated_at: datetime


class CreateTaskInput(BaseModel):
    workspace: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    details: str | None = None
    effort: int = Field(3, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    due_date: date | None = None


class CreateTaskOutput(BaseModel):
    task: Task
    dedupe_hint: str | None = None


class ListTasksInput(BaseModel):
    workspace: str = Field(..., min_length=1)
    include_done: bool = False


class ListTasksOutput(BaseModel):
    tasks: list[Task]
    total: int = Field(..., ge=0)
    pending: int = Field(..., ge=0)


class CompleteTaskInput(BaseModel):
    workspace: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)


class CompleteTaskOutput(BaseModel):
    task: Task


class TaskScore(BaseModel):
    task_id: str
    title: str
    score: int
    reasons: list[str] = Field(default_factory=list)


class ScoreTasksInput(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
    today: date


class ScoreTasksOutput(BaseModel):
    ranked: list[TaskScore] = Field(default_factory=list)


class BuildPlanInput(BaseModel):
    workspace: str = Field(..., min_length=1)
    today: date
    max_items: int = Field(3, ge=1, le=10)


class PlanItem(BaseModel):
    task_id: str
    title: str
    score: int
    reason: str


class BuildPlanOutput(BaseModel):
    plan: list[PlanItem] = Field(default_factory=list)
    total_pending: int = Field(..., ge=0)
    narrative: str


class SmartScheduleInput(BaseModel):
    workspace: str = Field(..., min_length=1)
    today: date
    user_text: str = Field(..., min_length=1)
    context_notes: list[str] = Field(default_factory=list)
    max_items: int = Field(5, ge=1, le=10)


class SmartScheduleOutput(BaseModel):
    status: Literal["needs_clarification", "scheduled"]
    assistant_message: str
    questions: list[str] = Field(default_factory=list)
    created_tasks: list[Task] = Field(default_factory=list)
    plan: list[PlanItem] = Field(default_factory=list)
    total_pending: int = Field(0, ge=0)
    narrative: str = ""
