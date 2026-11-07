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
PowerType= Literal["office", "military", "wealth", "influence", "legal_protection"]
Confidence = Literal["high","medium","low","unknown"]
DatePrecision = Literal["day", "month","year","relative", "unknown"]
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
    when: When = Field(default_factory=When)
    prov: proof
 
    @field_validator("participants")
    @classmethod
    def _no_duplicates(cls, v:list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("participants must be unique")
        return v
    
    @model_validator(mode="after")
    def _roles_are_participants(self)-> "HyperedgeBase":
        stray = set(self.roles) - set(self.participants)
        if stray:
            raise ValueError(f"roles given for non-participants: {sorted(stray)}")
        return self


##sub category
class Alliance(HyperedgeBase):
    type: Literal["alliance"] = "alliance"
    
class FamilyTie(HyperedgeBase):
    type: Literal["family_tie"] = "family_tie"
    
class CareerEvent(HyperedgeBase):
    type: Literal["career_event"] = "career_event"
    
#this is kinda similar to beef with certain thing and leading to two or more distinct side    
class Side(BaseModel):
    members: list[str] = Field(min_length=1, description="node ids")
    grievance: str #more so about why a has beef with b
    stated_vs_revealed: str | None = Field(default=None, description="what they say publically and what they actually do in real life")
    
class Rivalry(HyperedgeBase):
    """The Beef"""
    
    type: Literal["rivalry"] = "rivalry" # this is explaining the type fo the edge
    #explaing what side of the fued they belong to 
    sides: list[Side] = Field(min_length=2, description="the sides of the rivalry")
    cause_nodes: list[str] = Field(default_factory=list, description="the goal/value nodes that caused this rivalry")
    intensity: Literal["mild", "serious", "unknown"] = "unknown"
    status: Literal["active", "resolved", "dormant", "unknown"] = "unknown"
    
    @model_validator(mode="after")
    def _sides_are_participants(self) -> "Rivalry":
        in_sides = {m for s in self.sides for m in s.members}
        missing = in_sides - set(self.participants)
        if missing:
            raise ValueError(f"side members missing from participants: {sorted(missing)}")
        return self

## so thsi kinda like instantiation of the hyperedges class
Hyperedge = Annotated[
    Union[Alliance, FamilyTie, CareerEvent, Rivalry],
    Field(discriminator="type"),
]

# ----------------------------------------------------------------------------
# Graph: the whole thing. Checks that need to see ALL nodes and edges together.
# ----------------------------------------------------------------------------
class Graph(BaseModel):
    cutoff_year: int = CUTOFF_YEAR
    nodes: list[Node] = []
    edges: list[Hyperedge] = []
 
    @model_validator(mode="after")
    def _check_graph(self) -> "Graph":
        # Fast lookup table: node id -> node.
        by_id = {n.id: n for n in self.nodes}
        # No two nodes (or edges) may share an id.
        if len(by_id) != len(self.nodes):
            raise ValueError("duplicate node ids")
        if len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("duplicate edge ids")
 
        # Cutoff check on every node's date.
        for n in self.nodes:
            self._check_date(n.when, f"node {n.id}")
 
        for e in self.edges:
            # Every id an edge mentions must really exist (no dangling links).
            for pid in e.participants:
                if pid not in by_id:
                    raise ValueError(f"edge {e.id}: unknown participant '{pid}'")
            # Drivers must exist AND be motive nodes (goal or value), not people or events.
            for did in e.drivers:
                if did not in by_id:
                    raise ValueError(f"edge {e.id}: unknown driver '{did}'")
                if by_id[did].kind not in ("goal", "value"):
                    raise ValueError(f"edge {e.id}: driver '{did}' must be a goal or value node")
            # Rivalry-only: the cause nodes must exist too.
            if isinstance(e, Rivalry):
                for cid in e.cause_nodes:
                    if cid not in by_id:
                        raise ValueError(f"edge {e.id}: unknown cause node '{cid}'")
            # Cutoff check on every edge's date.
            self._check_date(e.when, f"edge {e.id}")
        return self
 
    def _check_date(self, w: When | None, label: str) -> None:
        # THE ANTI-LEAK RULE: reject anything dated after the cutoff year.
        if w is not None and w.year is not None and w.year > self.cutoff_year:
            raise ValueError(f"{label}: year {w.year} is after the cutoff {self.cutoff_year}")