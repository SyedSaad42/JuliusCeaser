"""
schema.py: the skeleton of the graph.    
"""
# hint type. refereing to be used later
from __future__ import annotations
# these all  for the node class and evaluation
from pydantic import BaseModel, Field, field_validator, model_validator
#extra meta data to a type
from typing import Literal, Annotated, Union

CUTOFF_YEAR = -45 # AS we dont intend on having context for any later than that 

# we need to create some classes for the sanity reasons
NodeKind= Literal["stakeholder", "person", "group","event", "power", "goal", "value"]
PowerType= Literal["office", " military", "wealth", "influence", "legal_protection"]
Confidence = Literal["high","medium","low","unknown"]
DatePrecision = Literal["day", "month","year","relative", "unkown"]
# we are using section for the resource reason as we have to build the neighbour nodes later
Section = Literal["political_position","family_ancestry","personal_ambitions","political_alliances","rivalries","military_influence","relationships","historical_actions"] 


#now we need to create When and proof class
class proof(BaseModel):
    """It give the associated link to the extract and concluded answer. Thus acting as a resource"""
     #source file obviously
    source_file: str = Field(description="e.g. 01_cassius.txt")
    from_section: Section | None = None
    supporting_quote: str = Field(min_length=8,max_length=200, description="Exact text copied from the source. Code later checks it appears word for word.")
    confidence: Confidence = "medium"
    
    
## now we need when class, which is not as important but it is necessary for the time check and event based context

class When(BaseModel):
    """A date that can be fuzzy. Kept separate so cutoff checks are simple."""
 
    year: int | None = Field(default=None, description="BCE negative, e.g. -48")
    month: int | None = Field(default=None, ge=1, le=12)
    precision: DatePrecision = "unknown"
    text: str | None = Field(default=None, description="Original wording, e.g. '48/47 BCE'")




    
    
#even though pydantic provides type validator but it only check one variable only ... but when
# you have a class filled up with attributes, we need a model_validator to for validating the type

@model_validator(mode="after") # so this gets called once the object is built
#we are calling the validating function on the object ***When****

def _precision_matches(self)-> "When":
    
    #if year not there and precision not is unknown that mean halucination
    # and thats exactly what we will kill
    
    if self.year is None and self.precision not in ("unknown","relative"):
        raise ValueError("a date with no year must have precision 'relative' or 'unknown'")
    #if the month is not none but the year exist.. haluucinationn
    if self.month is not None and self.year is None:
        raise ValueError("month given without a year")
    return self





    
#so we are creating our NODE with attributes
class Node(BaseModel):
    # ids are senstive thus will take a criterion based
    id:str = Field(pattern=r"^[a-z0-9_]+$")
    name:str #the name of the node
    kind: NodeKind 
    #this explain what kinda node this is , the stakeholder or  a associated group
    aliases: list[str] = [] 
    """ so the name of the charcter are not refering the injecteed question as juluis ceaser. instead just ceaser thus we need aliases """
    # rest two feature are mostly for other kinds 
    note: str | None = None
    when: When | None = None
    res: proof
    power_type: PowerType | None = None
    strength: Literal["low","medium", "high"] | None = None
    
    
    
    
#HYPEREDGE is the edge but in pydantic class form , consisting of much more context
class HyperedgeBase(BaseModel):
  #edges will have their own id , which will be whole context , perhaps in a nutshell
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    
    participants: list[str] = Field(min_length=2, description="node ids")
    
    roles: dict[str, str] = Field(default_factory=dict, description="node id -> role in this edge")
    stance: str | None = None
    drivers: list[str] = Field(default_factory=list, description="ids of goal/value nodes behind it")
    conditions: list[str] = Field(default_factory=list, description="what had to be true / what would change it")
    when: When
    prov: proof
 