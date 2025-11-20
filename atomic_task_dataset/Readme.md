# How to use the dataset
Use the following prompt to feed into LLMs. Insert question in corresponding fields (< question >).
```
You are an expert in multi-step logical and transitive reasoning. 
You will be given one multiple-choice question that already includes its answer options after the word "Options:". 
Use only the information in the question and no outside knowledge.

Background:
- These atomic tasks are designed to probe how specific attention heads contribute to different types of reasoning
  (e.g., scalar comparisons, temporal order, spatial relations, subset/implication, hierarchies, etc.).
- By measuring model performance on these tightly controlled tasks, we can infer which heads are most important for each reasoning pattern.
- Therefore, every example must be treated as a precise micro–reasoning problem: logically clean, unambiguous,
  and free from shortcuts or heuristics that allow guessing without real reasoning.

Question:
<question>

Instructions:
- Read the question and its options carefully.
- Treat the premises as the only ground truth; do not use any world knowledge or external facts.
- Use precise logical and transitive reasoning over the given information to determine which single option must be correct.
- If several options look plausible, choose the one that is strictly required by the premises.
- Your chosen answer must be exactly one of the options as written in the question.
- Do not include your reasoning or any extra text.

Output format:
- Answer only using this format:
- [ "answer": "<your chosen option, copied verbatim from the options>" ]
- Do not output anything before '[' or after ']'.

Your answer:
```


# Prompt we used to generate atomic data samples
We used ChatGPT-5.1 auto to generate synthetic data samples.

## General Instruction
```
You are an expert dataset creator for evaluating atomic reasoning abilities in large language models.

These atomic tasks will be used to probe how specific attention heads contribute to different types of reasoning. 
By measuring model performance on these controlled tasks, we can infer which heads are most important for each reasoning pattern.
Therefore, every example must be logically clean, unambiguous, and free from shortcuts that allow guessing without real reasoning.

Your goal in this run is to generate small, self-contained question–answer pairs for a single atomic reasoning task.
Each example must:

- Test exactly one clear reasoning pattern.
- Be solvable using only the information in the question (no external knowledge).
- Have a single, unambiguous correct answer.
- Be formulated as an EXPLICIT MULTIPLE-CHOICE question with clearly listed options.

------------------
TASK SCOPE (INPUT)
------------------

You will be given a TASK SCOPE block between the tags <TASK_SCOPE> and </TASK_SCOPE>.
This block defines the current atomic task:

- its task_type name,
- its semantic goal,
- the required style of question and answer,
- and 1–2 concrete examples.

You MUST strictly follow the TASK SCOPE when generating data.

<TASK_SCOPE>
[THE TASK-SPECIFIC INSTRUCTIONS GO HERE]
</TASK_SCOPE>

--------------------
GLOBAL OUTPUT RULES
--------------------

1. Output format
   - You must output a SINGLE valid JSON array.
   - The array must contain EXACTLY 20 objects.
   - Each object must have EXACTLY three string fields:
     - "question": the natural-language question, including the list of options at the end.
     - "answer": the correct answer in a canonical form, which MUST be exactly one of the options.
     - "task_type": the task type name (exactly as specified in the TASK SCOPE).
   - Example overall structure:
     [
       {"question": "... Options: [Alice, Bob, Carol].", "answer": "Alice", "task_type": "Transitive reasoning-scalar max"},
       ...
     ]
   - Do NOT output any explanations, comments, or markdown. Only output the JSON array.

2. Multiple-choice questions and options
   - Every question MUST be written as a multiple-choice question.
   - At the END of each question, append a clause that explicitly lists the options, using the word "Options:".
     - For example: "Options: Alice, Bob, Carol."
     - Or: "Options: \">\", \"<\", \"=\", \"unknown\"."
   - The "answer" field MUST be exactly one of these options (matching the text or symbol precisely).
   - All options must be of the same semantic type:
     - e.g., all person names, or all events, or all cities, or all short sentences.
   - The options should be reasonably hard to distinguish:
     - Do NOT include obviously irrelevant or absurd options.
     - Do NOT make one option trivially different in style or domain (e.g., three human names and one random object).

3. Diversity and style
   - Use diverse names and entities (people, cities, events, boxes, folders, animals, abstract labels like A/B/C, etc.).
   - Vary the wording and surface forms (e.g., taller/shorter, older/younger, before/after), 
     while preserving the same underlying logical pattern defined in the TASK SCOPE.
   - Avoid repeating the same sentence template with only names changed.
   - Vary the semantic domains where appropriate (not only height, but also age, weight, speed, prices, distances, times, etc., when allowed by the TASK SCOPE).

4. Logical correctness and cleanliness
   - All relations you use (taller than, older than, before, in, subset-of, manager-of, etc.) must be logically consistent and transitive where required.
   - Do NOT create contradictory statements (e.g., both A taller than B and B taller than A).
   - Do NOT create examples where the answer is underdetermined or where multiple answers could be correct.
   - Avoid tasks that require extra arithmetic, obscure world knowledge, or multiple different reasoning skills. Focus on the intended atomic pattern.
   - Each example should be solvable purely via the intended reasoning pattern described in the TASK SCOPE.

----------------------
FINAL INSTRUCTION
----------------------

Using ONLY the TASK SCOPE above and the global rules, generate EXACTLY 20 high-quality, diverse multiple-choice examples for that task.

Return them as a single valid JSON array of objects with fields:
"question", "answer", and "task_type".

Do NOT include any other text or formatting.
```


## Task Specific Instruction
### 1. Transitive reasoning-scalar max
```
<TASK_SCOPE>
task_type: "Transitive reasoning-scalar max"

Goal:
- Use scalar comparisons in natural language (height, age, weight, speed, wealth, intelligence, scores, distances, etc.).
- Ask for the maximum or minimum entity (e.g., tallest, oldest, youngest, heaviest, fastest).
- The question must end with a list of candidate entities as options.

Question requirements:
- In each question, mention between 3 and 6 entities:
  - 3–4 entities for easier to medium questions.
  - Occasionally 5–6 entities for harder questions.
- Describe their ordering via comparative adjectives. You do NOT need to compare every possible pair.
- Do NOT always use a simple sequential chain like "A > B, B > C":
  - Frequently use more interleaved comparisons, such as:
    - "A is shorter than B, C is taller than B, and D is shorter than A."
    - "A is older than B, C is younger than A, and D is older than C but younger than B."
  - It is fine if some questions are simple chains (for easier items), but many questions should involve criss-cross or partially overlapping comparisons.
- Examples of admissible adjectives:
  - taller / shorter
  - older / younger
  - heavier / lighter
  - faster / slower
  - richer / poorer
  - smarter / less intelligent
  - higher / lower score, longer / shorter distance, etc.
- Ensure there is a single, uniquely determined maximum or minimum among the options, even when the comparisons are not purely sequential and not all pairs are explicitly compared.
- All listed entities must be plausible candidates (no “none of the above” or extra distractors beyond the described entities).
- At the end of the question, append: "Options: [X, Y, Z]." (or more entities) listing exactly the involved entities (in any order).

Answer requirements:
- The answer must be exactly the NAME of the correct entity, matching one of the listed options.

Example 1 (do not copy verbatim):
- Question: "Alice is taller than Bob, and Bob is taller than Carol. Who is the tallest? Options: [Alice, Bob, Carol]."
- Answer: "Alice"

Example 2 (do not copy verbatim):
- Question: "Liam is shorter than Noah. Olivia is taller than Noah but shorter than Emma. Who is the tallest? Options: [Liam, Noah, Olivia, Emma]."
- Answer: "Emma"
</TASK_SCOPE>
```

### 2. Transitive reasoning-symbolic inequality
```
<TASK_SCOPE>
task_type: "Transitive reasoning-symbolic inequality"

Goal:
- Use symbolic inequalities like "A > B" instead of natural language.
- Ask for the relation between two variables, chosen from {">", "<", "=", "unknown"}.

Question requirements:
- In each question, involve between 3 and 6 distinct variables, such as A, B, C, D, E, F.
- Provide 2–5 inequality statements between these variables.
  - You do NOT need to compare every possible pair.
  - Do NOT always use a simple sequential chain like "A > B, B > C".
    - Frequently use more interleaved or criss-cross patterns, such as:
      - "A > B, C > B, D < A."
      - "A < B, C > A, D < C, E > B."
    - It is fine if some questions are simple chains (for easier items), but many questions should involve partially overlapping comparisons.
- After the inequalities, ask: "What is the relation between X and Y?" where X and Y are two of the involved variables.
- At the end, list the four possible relation symbols as options, always in the same set:
  - '>', '<', '=', 'unknown'
  - For example: "Options: [\">\", \"<\", \"=\", \"unknown\"]."
- Make sure to include a mix of cases where the correct answer is:
  - ">", "<", "=", and "unknown" — do NOT always make the answer ">". 
- Ensure that, given the inequalities and standard transitivity of ">" and "<":
  - The correct relation between X and Y is logically well-defined as exactly one of {">", "<", "=", "unknown"}.
  - For the "unknown" cases, it must be genuinely underdetermined (i.e., X could be greater, smaller, or equal depending on unspecified values).

Answer requirements:
- The answer must be exactly one of these four strings: ">", "<", "=", "unknown".

Example (do not copy verbatim):
- Question: "A > B, B > C. What is the relation between A and C? Options: [\">\", \"<\", \"=\", \"unknown\"]."
- Answer: ">"
</TASK_SCOPE>
```


### 3. Transitive reasoning-temporal order
```
<TASK_SCOPE>
task_type: "Transitive reasoning-temporal order"

Goal:
- Use temporal relations to define a partial order over events (who happens earlier / later).
- Temporal relations can be expressed via words ("before", "after") or via explicit time information 
  (clock times, dates, days of the week, durations).
- Ask which event is earliest or latest in time.
- Options are the candidate events.

Question requirements:
- Number of events:
  - In each question, you MUST introduce between 3 and 6 distinct named events.
  - Use 3–4 events for easier to medium questions.
  - Occasionally use 5–6 events for harder questions.
  - Do NOT use fewer than 3 or more than 6 events.
- Event names:
  - Use short, natural phrases such as "the meeting", "lunch", "the workshop", "the exam",
    "the quiz", "the interview", "the presentation", etc.
- Temporal cues:
  - Use one or more of the following temporal cues to relate the events:
    1) Explicit "before" / "after" relations.
       - e.g., "The meeting happens before lunch."
    2) Concrete time points on the same timeline.
       - e.g., "The meeting is at 9:00 AM, the workshop is at 11:30 AM."
    3) Relative times / durations.
       - e.g., "Lunch is 30 minutes after the meeting."
    4) Ordered calendar references when unambiguous.
       - e.g., "The lecture is on Monday, the quiz is on Wednesday, and the project review is on Friday."
  - You do NOT need to compare every possible pair of events.
- Structure:
  - Do NOT always use a simple sequential chain like 
    "Event A happens before event B. Event B happens before event C."
  - Frequently use more interleaved or criss-cross patterns, such as:
    - "The meeting happens before lunch. The workshop is at 3:00 PM, which is after lunch. 
       The exam is at 8:00 AM on the same day."
    - "The interview is on Tuesday morning at 10:00. The presentation is on Monday afternoon. 
       The quiz is 2 hours after the presentation."
  - It is fine if some questions are simple chains (for easier items), but many questions should 
    involve partially overlapping constraints and/or mixed temporal cues.
- Consistency:
  - All time expressions must refer to a single, coherent timeline (e.g., the same day or a clearly ordered set of days),
    so that the ordering is unambiguous.
  - Ensure that, given all the temporal information, there is a UNIQUE earliest event 
    (for "happens first" questions) or a UNIQUE latest event (for "happens last" questions).
- Question format:
  - Ask a question such as:
    - "Which event happens first?"
    - or "Which event happens last?"
- Options:
  - At the end, list exactly ALL involved events as options:
    - e.g., "Options: [the meeting, lunch, the workshop, the exam]."
  - The event names in the options must exactly match the names used in the description.

Answer requirements:
- The answer must be exactly the name/phrase of the correct event, matching one of the listed options.

Example (do not copy verbatim):
- Question: "The meeting happens before lunch. Lunch happens before the workshop. 
  Which event happens first? Options: [the meeting, lunch, the workshop]."
- Answer: "the meeting"
</TASK_SCOPE>
```


### 4. Transitive reasoning-spatial containment
```
<TASK_SCOPE>
task_type: "Transitive reasoning-spatial containment"

Goal:
- Use one-dimensional spatial relations or containment relations to define an order over objects/locations.
- Spatial relations can be expressed via:
  - directional words (left/right, north/south, above/below, closer/farther), or
  - explicit numeric distances / coordinates that imply order (meters, kilometers, floor numbers, etc.).
- Ask for an extreme location (leftmost/rightmost, northernmost/southernmost, highest/lowest, closest/farthest)
  or the ultimate / outermost container.
- Options are the candidate objects or locations.

Question requirements:
- Number of entities:
  - In each question, introduce between 3 and 6 distinct entities.
  - Use 3–4 entities for easier to medium questions.
  - Occasionally use 5–6 entities for harder questions.
  - Do NOT use fewer than 3 or more than 6 entities.
- Entity types:
  - Use diverse, everyday objects and locations, for example:
    - cities, towns, mountains, islands, planets, landmarks
    - boxes, drawers, bags, cabinets, shelves
    - cars in a parking lot, chairs in a row, books on a shelf, paintings on a wall
    - rooms, buildings, floors, subway stations, bus stops
  - Entities can be inside, outside, on top of, or next to larger structures, as long as the spatial relation is 1-D or nesting.
- Allowed spatial / containment cues:
  - Directional words:
    - north of / south of / east of / west of
    - to the left of / to the right of
    - above / below / over / under
    - closer to / farther from a reference point
  - Numeric positions and distances:
    - coordinates on one axis (e.g., positions on a line: "at 2 meters", "at 5 meters")
    - distances from a reference point (e.g., "5 km from the city center")
    - floor numbers or heights (e.g., "on the 3rd floor", "at a height of 10 meters")
  - Containment / outside relations:
    - in / inside / within / inside of
    - outside / outside of / not in
    - nested containers (e.g., "The coin is in the box, the box is in the drawer, the drawer is inside the cabinet.")
- Structure:
  - You do NOT need to compare every possible pair.
  - Do NOT always use a simple sequential chain like:
    - "City A is north of City B. City B is north of City C."
  - Frequently use more interleaved or criss-cross patterns and/or mixed cues, for example:
    - "Car A is parked 3 meters to the left of Car B. Car C is 2 meters to the right of Car B.
       Car D is 1 meter to the left of Car A."
    - "The toy is in Box A. Box A is on the second shelf. Box B is on the third shelf above Box A.
       The book is in Box B."
    - "Town X is 10 km north of City Y. Village Z is 5 km south of City Y.
       The lake is 15 km north of Town X."
  - It is fine if some questions are simple chains (for easier items), but many questions should involve
    partially overlapping constraints or a mix of word-based relations and numeric distances.
- Consistency:
  - All spatial information must be consistent with a single one-dimensional ordering (e.g., along a line, north–south direction, vertical height, distance from a reference point) or a clear containment hierarchy.
  - Ensure that, given all information, there is a UNIQUE:
    - extreme position (leftmost/rightmost/northernmost/southernmost/highest/lowest/closest/farthest), or
    - ultimate / outermost location (e.g., the outermost container or final place where an object ends up).
- Question format:
  - Ask a question such as:
    - "Which object is the leftmost?"
    - "Which city is the southernmost?"
    - "Which item is highest?"
    - "Which car is closest to the entrance?"
    - "Where is the coin ultimately located?"
- Options:
  - At the end, list exactly ALL involved entities as options:
    - e.g., "Options: [City A, City B, City C, City D]."
  - The entity names in the options must exactly match the names used in the description.
  - Do NOT add extra “none of the above” options.

Answer requirements:
- The answer must be exactly one of the listed entities or location phrases.

Example (do not copy verbatim):
- Question: "City A is north of City B. City B is north of City C. Which city is the southernmost?
  Options: [City A, City B, City C]."
- Answer: "City C"
</TASK_SCOPE>
```


### 5. Transitive reasoning-subset/implication
```
<TASK_SCOPE>
task_type: "Transitive reasoning-subset/implication"

Goal:
- Test reasoning over class inclusion and logical implication.
- Premises should express that one condition/category guarantees or excludes another
  (e.g., subset, superset, incompatibility), and the question should require chaining
  2–4 such premises.
- Questions should ask whether a conclusion MUST be true, or which conclusion MUST be true.
- Always present possible answers as options.

Question requirements:

- Premises:
  - Use between 2 and 4 premise sentences.
  - Involve between 3 and 6 distinct categories / properties
    (e.g., students, doctors, scientists, musicians; “likes math”, “owns a car”, “has a degree”).
  - Allowed logical forms include, but are not limited to:
    1) Universal inclusion:
       - "All X are Y."
       - "Every X is a Y."
       - "Any X is also a Y."
       - "Whoever is X is Y."
       - "Anyone who is X must be Y."
       - "Being X guarantees being Y."
       - "To be X, you have to be Y."
    2) Conditional implication:
       - "If someone is X, then they are Y."
       - "If a person has property P, then they also have property Q."
       - "Whenever X happens, Y happens."
       - "X always leads to Y."
    3) Negative / exclusion relations:
       - "No X are Y."
       - "Nobody who is X is Y."
       - "If someone is X, then they are not Y."
       - "Being X excludes being Y."
    4) Paraphrases of inclusion / subset:
       - "Every pianist is a kind of musician."
       - "All surgeons are a subset of doctors."
       - "All roses belong to the group of flowers."
       - "Every biology major is included in the science students."
  - You do NOT need to mention every relation between all pairs of categories.
  - Avoid probabilistic or non-logical phrases such as "usually", "often", "can", "might", "sometimes":
    the premises should describe strict, logical rules (always true).

- Question styles:
  There are TWO allowed styles, and both should be used across the dataset.

  (a) Yes/No entailment:
      - After the premises, present a single candidate conclusion sentence.
      - Ask whether this conclusion MUST be true given the premises.
      - Example pattern (do not copy verbatim):
        - "Everyone who is a pianist is a musician.
           Every musician enjoys music.
           Conclusion: Everyone who is a pianist enjoys music.
           Is this conclusion necessarily true?"
      - Options must be exactly: "[Yes, No]."
      - Include both:
        - cases where the conclusion is entailed, and
        - cases where the conclusion is NOT entailed (false or underdetermined).

  (b) Conclusion selection:
      - After the premises, ask which conclusion MUST be true.
        - e.g., "Which of the following statements must be true?"
      - Provide 3–4 candidate conclusion sentences, only ONE of which is logically correct (entailed).
      - Wrong options must be plausible-sounding but logically incorrect or not guaranteed.
      - Do NOT use "None of the above" or "All of the above" as options.
      - Example pattern (do not copy verbatim):
        - "All surgeons are doctors.
           All doctors have medical training.
           Which statement must be true?
           Options: [All surgeons have medical training.;
                     Some surgeons have no medical training.;
                     All people with medical training are surgeons.]."

- Variety requirements:
  - Mix different surface forms for the same logical pattern:
    - "All X are Y" / "Every X is a Y" / "Anyone who is X must be Y" / "If someone is X, then they are Y".
  - Include both positive chains and chains with negation:
    - e.g., "All X are Y. No Y are Z." → "No X are Z."
    - e.g., "If someone is X then they are Y. Nobody who is Y is Z."
  - Vary the type of conclusion:
    - universal claims ("Every X has property Y").
    - existence of counterexamples ("Some X are not Y").
    - implications ("If someone is X, then they must be Z.").
  - Include cases where chaining 3 or more premises is required, and cases where the conclusion
    almost follows but fails (one missing or reversed link).

Answer requirements:
- For Yes/No questions:
  - The answer must be exactly "Yes" or "No".
- For conclusion-selection questions:
  - The answer must be exactly one of the candidate conclusion sentences, copied verbatim from the options.

Example 1 (do not copy verbatim):
- Question: "All sparrows are birds. All birds are animals.
            Is every sparrow an animal? Options: [Yes, No]."
- Answer: "Yes"

Example 2 (do not copy verbatim):
- Question: "Every mathematician is a scientist. Every scientist has a degree.
            What can we say about every mathematician?
            Options: [Every mathematician has a degree.;
                      Every mathematician is an engineer.;
                      Some mathematicians have no degree.]."
- Answer: "Every mathematician has a degree."
</TASK_SCOPE>
```


### 6. Transitive reasoning-hierarchy
```
<TASK_SCOPE>
task_type: "Transitive reasoning-hierarchy"

Goal:
- Test reasoning over hierarchical, transitive relations (who is above whom in a tree / hierarchy).
- Use relations like:
  - organizational chains (manager, supervisor, director, CEO),
  - family trees (parent, grandparent, ancestor),
  - rank hierarchies (junior/senior, lower/higher level),
  - category hierarchies (subtype / supertype, subcategory / category).
- Ask for a higher-level ancestor / manager / superior / more general category.
- Options are candidate people, nodes, or categories.

Question requirements:

- Hierarchy type (pick ONE domain per question):
  - Organizational hierarchy:
    - employees, team leads, managers, directors, CEOs, etc.
  - Family / ancestry:
    - parents, children, grandparents, ancestors, etc.
  - Rank / role hierarchy:
    - junior vs. senior positions, lower vs. higher levels.
  - Category / taxonomy:
    - species / genus / family, file / folder / drive, team / division / organization, etc.

- Number of nodes:
  - In each question, introduce between 3 and 6 distinct nodes (people, roles, or categories).
  - 3–4 nodes for easier to medium questions.
  - Occasionally 5–6 nodes for harder questions.
  - Do NOT use fewer than 3 or more than 6 nodes.

- Allowed relational phrases (all must encode a clear “higher than / lower than” relation):
  - Organizational:
    - "X manages Y.", "Y reports to X."
    - "X is Y’s manager / supervisor / team lead / director / boss."
    - "Y works under X.", "Y is below X in the hierarchy."
  - Family:
    - "X is Y’s parent / father / mother.", "Y is X’s child / son / daughter."
    - "X is Y’s grandparent.", "X is an ancestor of Y.", "Y is a descendant of X."
  - Rank:
    - "X is above Y in rank.", "Y is below X in rank."
    - "X holds a higher position than Y.", "Y has a lower rank than X."
  - Category:
    - "X is a subtype of Y.", "X is a kind of Y."
    - "Y is a broader category than X.", "Y is the more general category that includes X."
    - "X belongs to Y, and Y belongs to Z."

- Hierarchy structure:
  - Use 2–5 explicit relational statements to connect the nodes.
  - The implied structure should form a **tree or simple hierarchy**:
    - no cycles (no A above B and B above A),
    - relations are transitive (if A above B and B above C, then A above C).
  - Do NOT always use a trivial straight chain:
    - Avoid only "A manages B. B manages C." in most questions.
  - Frequently create slightly richer structures, such as:
    - branching:
      - "Alex manages Ben and Carla. Carla manages Diego."
    - longer chains:
      - "A is above B. B is above C. C is above D."
    - mixed directions:
      - "Ben reports to Alex. Carla reports to Ben. Diego reports to Carla."

- Query type (what you ask about):
  - The question should require at least one transitive step (often two or more).
  - Allowed query patterns include:
    - Ask for a specific ancestor / manager:
      - "Who is X’s manager’s manager?"
      - "Who is X’s top-level manager?"
      - "Who is X’s grandparent?"
      - "Who is the earliest ancestor of X mentioned here?"
    - Ask for the highest node in a local subtree:
      - "Who is the highest-ranking person among these people?"
      - "Which role is at the top of this team’s hierarchy?"
    - Ask for the most general category:
      - "Which category is the most general category that still contains X?"
  - Avoid questions that can be answered without using transitivity
    (e.g., directly stated one-step relations only).

- Options:
  - At the end of the question, list several nodes as options:
    - e.g., "Options: [Alice, Ben, Carla, Diego]."
  - The options should:
    - include the correct answer,
    - usually include most or all nodes mentioned in the premises,
    - NOT include meta-options like "none of the above".
  - The names / labels in the options must **exactly** match those used in the description.

- Uniqueness:
  - Ensure the requested node is uniquely determined by the hierarchy:
    - exactly one correct ancestor / manager / superior / super-category that satisfies the question.
  - Do NOT create ambiguous cases where more than one option could be considered correct.

Answer requirements:
- The answer must be exactly the correct node’s name/label, copied verbatim from the options.

Example (do not copy verbatim):
- Question:
  - "Alex manages Beth and Carlos. Beth supervises Diana. Diana manages Ethan. Carlos reports directly to Alex.
  Who is Ethan's top-level manager in this hierarchy?
  Options: [Alex, Beth, Carlos, Diana, Ethan]."
- Answer:
  - "ALex"
</TASK_SCOPE>
```



