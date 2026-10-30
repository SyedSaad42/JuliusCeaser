# Drama

An in-progress simulation of the characters and relationships in Shakespeare's
*Julius Caesar*. The project aims to explore how alliances, rivalries, and
political ties shift throughout the play, using [HyperNetX](https://hypernetx.readthedocs.io/)
to represent relationships that involve more than two characters at a time.

## Why hypergraphs?

Many important moments in *Julius Caesar* involve groups: conspirators plan
together, factions form, and public events bring multiple characters into
conflict. A hypergraph can represent each character as a node and each shared
event or relationship as a hyperedge connecting all the characters involved.
This makes it possible to explore the play's dense social structure beyond
simple pairwise connections.

## Project goals

- Represent characters and their relationships as a hypergraph.
- Model how those relationships change as the play unfolds.
- Explore the characters, groups, and connections in the resulting network.

## Status

Early development. The simulation and its data model are still being built.

## Technology

- [Python](https://www.python.org/)
- [HyperNetX](https://hypernetx.readthedocs.io/)

