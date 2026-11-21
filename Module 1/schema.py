from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    topic: str = Field(..., description="The topic for which to generate text.",example="Neural Network")
    course: str = Field(..., description="The course for which to generate text.", example="Artificial Intelligence")
    student_level: str = Field(..., description="The educational level of the student.", example="beginner")