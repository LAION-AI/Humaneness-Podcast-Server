"""Public script contract. No file paths, URLs, code, or shell commands in requests."""
from typing import Literal, Annotated
import re
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)

class Delivery(Strict):
    text: str=Field(min_length=1,max_length=600)
    cue: str=Field(default='easy, spontaneous, gently spoken',min_length=1,max_length=240)
    @field_validator('cue')
    @classmethod
    def cue_syntax(cls,v):
        if any(x in v for x in ('(',')','\n','GENERAL:','SCRIPT:')): raise ValueError('Cue must be a short English direction without parentheses or prompt headers')
        return v

class Speech(Strict):
    type: Literal['speech']='speech'
    speaker: str=Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,31}$')
    text: str=Field(min_length=1,max_length=600)
    direction: str=Field(default='warm and sincere; relaxed, spontaneous, gently spoken',max_length=240)
    delivery_chunks: list[Delivery] | None=Field(default=None,min_length=1,max_length=2)
    pause_after_ms: int=Field(default=400,ge=0,le=10000)
    @model_validator(mode='after')
    def check(self):
        if len(re.findall(r'\S+',self.text))>60:raise ValueError('Split speech into turns of at most 60 words (25.6-second model ceiling)')
        if '(' in self.text or ')' in self.text:raise ValueError('Put English parenthetical directions in direction/delivery_chunks; text contains only spoken words')
        Delivery.cue_syntax(self.direction)
        if self.delivery_chunks and ' '.join(x.text for x in self.delivery_chunks)!=self.text:raise ValueError('delivery_chunks texts joined by one space must exactly equal text')
        return self

class Pause(Strict):
    type: Literal['pause']
    duration_ms: int=Field(ge=0,le=10000)

class Options(Strict):
    best_of: int=Field(default=4,ge=1,le=16)
    seed: int=Field(default=20261109,ge=0,le=2147483647)
    temperature: float=Field(default=1.0,ge=0,le=1.5)
    cfg: float=Field(default=2.5,ge=1,le=4)
    @field_validator('best_of')
    @classmethod
    def discrete_n(cls,v):
        if v not in (1,2,4,8,16):raise ValueError('best_of must be 1, 2, 4, 8 or 16')
        return v

class Episode(Strict):
    version: Literal['1.0']='1.0'
    title: str=Field(default='Untitled podcast',min_length=1,max_length=160)
    language: Literal['en','de','nl','fr','es']
    speakers: dict[str,str]=Field(min_length=1,max_length=8,description='Speaker ID -> server reference voice ID')
    turns: list[Annotated[Speech | Pause,Field(discriminator='type')]]=Field(min_length=1,max_length=600)
    options: Options=Field(default_factory=Options)
    @model_validator(mode='after')
    def check_episode(self):
        speech=[t for t in self.turns if isinstance(t,Speech)]
        if not speech:raise ValueError('At least one speech turn is required')
        if any(t.speaker not in self.speakers for t in speech):raise ValueError('Every speaker must be declared in speakers')
        if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}',s) for s in self.speakers):raise ValueError('Invalid speaker ID')
        if sum(len(t.text) for t in speech)>60000:raise ValueError('Maximum script size is 60,000 characters')
        seconds=sum(len(t.text.split())/2.2+t.pause_after_ms/1000 for t in speech)+sum(t.duration_ms/1000 for t in self.turns if isinstance(t,Pause))
        if seconds>3600:raise ValueError('Estimated episode duration exceeds one hour; split into jobs')
        return self
