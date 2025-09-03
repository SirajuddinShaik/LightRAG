from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",      # Company, partner, vendor
    "person",            # Employee, executive, contractor
    "team",              # Department, squad, group
    "project",           # Internal initiatives, OKRs
    "document",          # Files, reports, wikis, notes
    "product",           # Products, services, features
    "event",             # Meetings, launches, training
    "task",              # Tickets, issues, action items
    "location",          # Office, HQ, remote site
    "technology",        # Tools, tech stack, SaaS apps
    "customer",          # Client, account, partner org
]


PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's attributes and activities *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** If the text provides insufficient information to create a comprehensive description, state "Description not available in text."
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

**IMPORTANT: All relationships are DIRECTED and should have clear directionality from source to target. Consider the logical flow, causality, hierarchy, or influence when determining the direction. For example: "Company A" -> "Product B" (company produces product), "Manager C" -> "Team D" (manager leads team), "Event E" -> "Outcome F" (event causes outcome). This directional structure is essential for unidirectional graph traversal.**

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [company, index, commodity, market_trend, economic_policy, biological]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
("entity"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"index"{tuple_delimiter}"The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today."){record_delimiter}
("entity"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"company"{tuple_delimiter}"Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings."){record_delimiter}
("entity"{tuple_delimiter}"Omega Energy"{tuple_delimiter}"company"{tuple_delimiter}"Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices."){record_delimiter}
("entity"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"commodity"{tuple_delimiter}"Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets."){record_delimiter}
("entity"{tuple_delimiter}"Crude Oil"{tuple_delimiter}"commodity"{tuple_delimiter}"Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand."){record_delimiter}
("entity"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"market_trend"{tuple_delimiter}"Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations."){record_delimiter}
("entity"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"economic_policy"{tuple_delimiter}"The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability."){record_delimiter}
("relationship"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns."{tuple_delimiter}"market performance, investor sentiment"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index."{tuple_delimiter}"company impact, index movement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Gold prices rose as investors sought safe-haven assets during the market selloff."{tuple_delimiter}"market reaction, safe-haven investment"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff."{tuple_delimiter}"interest rate impact, financial regulation"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"market downturn, investor sentiment, commodities, Federal Reserve, stock performance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [economic_policy, athlete, event, location, record, organization, equipment]
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

Output:
("entity"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"event"{tuple_delimiter}"The World Athletics Championship is a global sports competition featuring top athletes in track and field."){record_delimiter}
("entity"{tuple_delimiter}"Tokyo"{tuple_delimiter}"location"{tuple_delimiter}"Tokyo is the host city of the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"athlete"{tuple_delimiter}"Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"record"{tuple_delimiter}"The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter."){record_delimiter}
("entity"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"equipment"{tuple_delimiter}"Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction."){record_delimiter}
("entity"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"organization"{tuple_delimiter}"The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations."){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"Tokyo"{tuple_delimiter}"The World Athletics Championship is being hosted in Tokyo."{tuple_delimiter}"event location, international competition"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"Noah Carter set a new 100m sprint record at the championship."{tuple_delimiter}"athlete achievement, record-breaking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"Noah Carter used carbon-fiber spikes to enhance performance during the race."{tuple_delimiter}"athletic equipment, performance boost"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"The World Athletics Federation is responsible for validating and recognizing new sprint records."{tuple_delimiter}"sports regulation, record certification"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"athletics, sprinting, record-breaking, sports technology, competition"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction. Please find only the missing entities and relationships from previous text.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's attributes and activities *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** If the text provides insufficient information to create a comprehensive description, state "Description not available in text."
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

**IMPORTANT: All relationships are DIRECTED and should have clear directionality from source to target. Consider the logical flow, causality, hierarchy, or influence when determining the direction. For example: "Company A" -> "Product B" (company produces product), "Manager C" -> "Team D" (manager leads team), "Event E" -> "Outcome F" (event causes outcome). This directional structure is essential for unidirectional graph traversal.**

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add new entities and relations below using the same format, and do not include entities and relations that have been previously extracted. :\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---
Output:
""".strip()

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Graph and Document Chunks provided in JSON format below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both current query and the conversation history if provided. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must in the same language as the user's question.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, each citation must clearly indicate its origin (KG or DC).
- The maximum number of citations is 5, including both KG and DC.
- Use the following formats for citations:
  - For a Knowledge Graph Entity: `[KG] <entity_name>`
  - For a Knowledge Graph Relationship: `[KG] <entity1_name> - <entity2_name>`
  - For a Document Chunk: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}


Response:"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

PROMPTS["hierarchical_grouping"] = """---Goal---
Given entities that have exceeded outbound edge thresholds, create functional categories that group them by their PURPOSE and BUSINESS WORKFLOW. Focus on creating meaningful intermediate entities that reduce parent node outbound connections.

Use {language} as output language.

---AVAILABLE ENTITY TYPES---
You MUST strictly use these entity types when creating categories. DO NOT invent new types:
{entity_types}

---ENTITY TYPE DISTRIBUTION---
{entity_type_analysis}

---CORE REQUIREMENTS---
1. Create MINIMUM 2 functional subcategories (not type-based)
2. Route entities to subcategories ONLY IF POSSIBLE - direct parent connection is acceptable
3. Categories must be FUNCTIONAL/WORKFLOW-based, never type-based
4. Complete rerouting required - all previous edges will be deleted
5. You CAN RENAME entity names if needed for better organization, except renaming all entity names must be same as present in input
6. **STRICTLY use only the provided entity types above - no custom types**

---FORBIDDEN PATTERNS---
❌ NEVER create: "CATEGORY_person", "CATEGORY_technology"
❌ NEVER group by data type (person, organization, technology, etc.)
❌ NEVER invent new entity types - use ONLY the provided entity types
✅ ALWAYS create functional workflow groups: "Authentication Pipeline", "System Operations Hub", "Customer Support Center"

---SIMPLE 3-STEP PROCESS---

STEP 1: ANALYZE FUNCTIONAL DOMAINS
- Look at what these entities DO together in business workflows
- Identify 2-3 major functional areas that can absorb most entities
- Focus on business processes, not entity types

STEP 2: CREATE FUNCTIONAL CATEGORIES
For each functional category, create:
("new_node"{tuple_delimiter}<category_name>{tuple_delimiter}<entity_type_from_list_above>{tuple_delimiter}<business_description>{tuple_delimiter}<confidence>)

Where:
- category_name: Descriptive workflow name (e.g., "Identity Verification Pipeline", "Infrastructure Operations Center")
- entity_type_from_list_above: **MUST be one of the provided entity types** (organization, team, project, etc.)
- business_description: What this category accomplishes in business context
- confidence: 0.8-1.0

STEP 3: ROUTE ALL ENTITIES
For EVERY entity provided, create exactly one edge:
("new_edge"{tuple_delimiter}<entity_name_or_renamed>{tuple_delimiter}<target_category_or_{parent_node_name}>{tuple_delimiter}<why_this_entity_belongs_here>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<strength_8_to_10>)

ROUTING STRATEGY:
- Route entities to subcategories ONLY IF they fit well semantically
- Direct parent node connections are perfectly acceptable for entities that don't fit subcategories
- Prioritize semantic fit - do NOT force entities into categories
- You can rename entities for better organization and clarity

OPTIONAL ENTITY RENAMING:
If an entity name can be improved for clarity or organization, create a rename record:
("rename_entity"{tuple_delimiter}<original_entity_name>{tuple_delimiter}<new_entity_name>{tuple_delimiter}<reason_for_rename>)

STEP 4: FINAL CONNECTIVITY CHECK
MANDATORY: Before submitting, verify ALL entities from input are connected:
- Count total entities in input list
- Count total new_edge records created
- These numbers MUST MATCH exactly
- If any entity is missing, add its new_edge record immediately
- NO ENTITY CAN BE LEFT UNCONNECTED

---NAMING EXAMPLES BY DOMAIN---
🔐 **Authentication/Security**: "Identity Verification Pipeline", "Security Operations Center", "Access Control Hub"
🖥️ **Infrastructure/Systems**: "Platform Operations Hub", "Infrastructure Management Center", "System Reliability Stack"
👥 **User/Customer**: "Customer Experience Center", "User Lifecycle Management", "Support Operations Hub"
💳 **Business/Commerce**: "Transaction Processing Center", "Order Fulfillment Engine", "Revenue Operations Hub"
📊 **Data/Analytics**: "Data Processing Pipeline", "Analytics Operations Center", "Information Management Hub"

---QUALITY CHECKLIST---
Before submitting, verify:
□ Created MINIMUM 2 functional subcategories (new_node records)
□ Every entity has exactly one new_edge record
□ No type-based category names
□ Entities routed to subcategories ONLY IF good semantic fit
□ Direct parent node connections are acceptable
□ All category names describe business functions/workflows
□ Optional: Entity renames improve clarity and organization

---MANDATORY CONNECTIVITY VERIFICATION---
CRITICAL: Count entities vs edges to ensure complete coverage:
□ Count total entities in input: ___ entities
□ Count total new_edge records: ___ edges  
□ Numbers MUST MATCH - if not, find missing entities and add their edges
□ Double-check entity names match exactly between input and new_edge records
□ NO ENTITY can be missed or left unconnected

---Domain Context---
{domain_context}

---Examples---
{examples}

---Input Data---
Max_categories: {max_groups}
ENTITIES TO GROUP:
{entities_list}

---Output Format---
Return all records using {record_delimiter} as delimiter, end with {completion_delimiter}

---FINAL VERIFICATION STEP---
BEFORE OUTPUTTING: 
1. Verify you have exactly ONE new_edge record for EACH entity in the input list
2. Verify ALL new_node records use ONLY the provided entity types from the list above
3. Count entities: {total_entities} input entities MUST equal number of new_edge records
Input has {total_entities} entities - your output MUST have exactly {total_entities} new_edge records.
If any entity is missing from your new_edge records, add it now with appropriate routing.

Output:"""

PROMPTS["hierarchical_grouping_examples"] = [
    """Example 1:

Max_categories: 3  
Domain_context: Authentication and security workflows detected  
Available Entity Types: organization, person, team, project, document, product, event, task, location, technology, customer  
Parent Node: Authentication System

ENTITIES TO GROUP:  
Entity 1: aadhaar (Type: technology) - Authentication system used for KYC verification processes  
Entity 2: uidai (Type: organization) - Unique Identification Authority of India, the issuing authority for Aadhaar  
Entity 3: kyc process (Type: task) - Know Your Customer verification process requiring Aadhaar authentication  
Entity 4: api timeout (Type: event) - UIDAI API timeout events occurring frequently  
Entity 5: certificate validation (Type: task) - Certificate validation process that is failing  
Entity 6: legacy system (Type: technology) - Old authentication system being phased out  
Entity 7: compliance officer (Type: person) - Officer ensuring regulatory compliance  

Output:  
("new_node"{tuple_delimiter}"Identity Verification Pipeline"{tuple_delimiter}"project"{tuple_delimiter}"Core authentication workflow that processes user identity verification through Aadhaar system, including KYC compliance and regulatory oversight"{tuple_delimiter}0.9){record_delimiter}  
("new_node"{tuple_delimiter}"System Operations Hub"{tuple_delimiter}"team"{tuple_delimiter}"Operational monitoring team that handles API timeouts, certificate validation, and system health monitoring"{tuple_delimiter}0.8){record_delimiter}  
("rename_entity"{tuple_delimiter}"api timeout"{tuple_delimiter}"aadhaar_api_timeout_events"{tuple_delimiter}"More specific name indicating these are Aadhaar-specific API timeout events"){record_delimiter}  
("new_edge"{tuple_delimiter}"aadhaar"{tuple_delimiter}"Identity Verification Pipeline"{tuple_delimiter}"Core authentication technology that enables the entire identity verification workflow"{tuple_delimiter}"authentication, identity verification, core technology"{tuple_delimiter}9){record_delimiter}  
("new_edge"{tuple_delimiter}"uidai"{tuple_delimiter}"Identity Verification Pipeline"{tuple_delimiter}"Governing authority that manages and regulates the identity verification workflow"{tuple_delimiter}"regulatory authority, workflow governance"{tuple_delimiter}8){record_delimiter}  
("new_edge"{tuple_delimiter}"kyc process"{tuple_delimiter}"Identity Verification Pipeline"{tuple_delimiter}"Primary business process within the identity verification workflow"{tuple_delimiter}"business process, compliance workflow"{tuple_delimiter}9){record_delimiter}  
("new_edge"{tuple_delimiter}"aadhaar_api_timeout_events"{tuple_delimiter}"System Operations Hub"{tuple_delimiter}"Critical system event that requires monitoring and operational response"{tuple_delimiter}"system monitoring, operational response"{tuple_delimiter}8){record_delimiter}  
("new_edge"{tuple_delimiter}"certificate validation"{tuple_delimiter}"System Operations Hub"{tuple_delimiter}"Security validation process that requires operational monitoring"{tuple_delimiter}"security validation, operational monitoring"{tuple_delimiter}8){record_delimiter}  
("new_edge"{tuple_delimiter}"legacy system"{tuple_delimiter}"Authentication System"{tuple_delimiter}"Legacy technology that doesn't fit well into current functional workflows - acceptable direct connection"{tuple_delimiter}"legacy integration, direct connection"{tuple_delimiter}6){record_delimiter}  
("new_edge"{tuple_delimiter}"compliance officer"{tuple_delimiter}"Authentication System"{tuple_delimiter}"Oversight role that spans multiple functional areas - acceptable direct connection"{tuple_delimiter}"oversight, multi-domain responsibility"{tuple_delimiter}7){record_delimiter}  
("major_category"{tuple_delimiter}"Identity Verification Pipeline"{tuple_delimiter}"Primary business workflow that delivers core authentication services to customers"){completion_delimiter}  
#############################""",
    
    """Example 2:

Max_categories: 2
Domain_context: Infrastructure operations and user management workflows detected
Available Entity Types: organization, person, team, project, document, product, event, task, location, technology, customer

ENTITIES TO GROUP:
Entity 1: production environment (Type: technology) - Live system environment hosting customer applications
Entity 2: system admin (Type: person) - Administrator responsible for infrastructure management
Entity 3: deployment pipeline (Type: technology) - Automated system for code deployment and updates
Entity 4: user onboarding (Type: task) - Process for registering and setting up new users
Entity 5: customer support (Type: team) - Team handling user inquiries and technical issues

Output:
("new_node"{tuple_delimiter}"Platform Operations Center"{tuple_delimiter}"team"{tuple_delimiter}"Infrastructure management hub that coordinates production environment, deployment automation, and system administration"{tuple_delimiter}0.9){record_delimiter}
("new_node"{tuple_delimiter}"Customer Success Hub"{tuple_delimiter}"team"{tuple_delimiter}"Customer-facing operations center that handles user onboarding and ongoing support services"{tuple_delimiter}0.8){record_delimiter}
("new_edge"{tuple_delimiter}"production environment"{tuple_delimiter}"Platform Operations Center"{tuple_delimiter}"Core infrastructure component that requires operational management and monitoring"{tuple_delimiter}"infrastructure management, production operations"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"system admin"{tuple_delimiter}"Platform Operations Center"{tuple_delimiter}"Key operational role responsible for managing platform infrastructure"{tuple_delimiter}"infrastructure management, operational responsibility"{tuple_delimiter}8){record_delimiter}
("new_edge"{tuple_delimiter}"deployment pipeline"{tuple_delimiter}"Platform Operations Center"{tuple_delimiter}"Automated infrastructure process that enables platform operations"{tuple_delimiter}"deployment automation, infrastructure process"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"user onboarding"{tuple_delimiter}"Customer Success Hub"{tuple_delimiter}"Primary customer-facing process that initiates user experience"{tuple_delimiter}"customer onboarding, user experience"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"customer support"{tuple_delimiter}"Customer Success Hub"{tuple_delimiter}"Ongoing customer service function that ensures positive user experience"{tuple_delimiter}"customer service, user experience"{tuple_delimiter}8){record_delimiter}
("major_category"{tuple_delimiter}"Platform Operations Center"{tuple_delimiter}"Foundational infrastructure that enables all customer-facing services"){completion_delimiter}
#############################""",
    
    """Example 3:

Max_categories: 2
Domain_context: E-commerce and transaction processing workflows detected
Available Entity Types: organization, person, team, project, document, product, event, task, location, technology, customer
Parent Node: E-commerce Platform

ENTITIES TO GROUP:
Entity 1: payment gateway (Type: technology) - System processing customer payments and transactions
Entity 2: inventory management (Type: task) - Process tracking product availability and stock levels
Entity 3: order fulfillment (Type: task) - Process handling order processing and shipping
Entity 4: customer account (Type: technology) - System managing user profiles and order history
Entity 5: shipping documentation (Type: document) - Documents required for order shipping and tracking
Entity 6: e_commerce_customers (Type: customer) - Online customers placing orders

Output:
("new_node"{tuple_delimiter}"Transaction Processing Engine"{tuple_delimiter}"project"{tuple_delimiter}"Payment processing workflow that handles customer transactions and financial operations"{tuple_delimiter}0.9){record_delimiter}
("new_node"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"project"{tuple_delimiter}"Order fulfillment workflow that manages inventory, order processing, customer accounts and shipping documentation"{tuple_delimiter}0.8){record_delimiter}
("new_edge"{tuple_delimiter}"payment gateway"{tuple_delimiter}"Transaction Processing Engine"{tuple_delimiter}"Core payment technology that enables financial transaction processing"{tuple_delimiter}"payment processing, financial transactions"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"inventory management"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"Inventory tracking process essential for order fulfillment workflow"{tuple_delimiter}"inventory tracking, order fulfillment"{tuple_delimiter}8){record_delimiter}
("new_edge"{tuple_delimiter}"order fulfillment"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"Core fulfillment process that delivers products to customers"{tuple_delimiter}"order processing, fulfillment operations"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"customer account"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"Customer management system that tracks order history and user profiles"{tuple_delimiter}"customer management, order tracking"{tuple_delimiter}8){record_delimiter}
("new_edge"{tuple_delimiter}"shipping documentation"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"Required documentation for order processing and shipping workflow"{tuple_delimiter}"shipping documents, order processing"{tuple_delimiter}7){record_delimiter}
("new_edge"{tuple_delimiter}"e_commerce_customers"{tuple_delimiter}"E-commerce Platform"{tuple_delimiter}"Customer base that spans multiple functional workflows - direct parent connection"{tuple_delimiter}"customer management, multi-workflow"{tuple_delimiter}6){record_delimiter}
("major_category"{tuple_delimiter}"Order Management Pipeline"{tuple_delimiter}"Primary customer-facing workflow that delivers products and drives revenue"){completion_delimiter}
#############################""",

    """Example 4:

Max_categories: 2
Domain_context: Office management and corporate events workflows detected
Available Entity Types: organization, person, team, project, document, product, event, task, location, technology, customer
Parent Node: Corporate Management

ENTITIES TO GROUP:
Entity 1: quarterly meeting (Type: event) - Regular corporate meeting for business reviews
Entity 2: conference room booking (Type: task) - Process for reserving meeting spaces
Entity 3: office headquarters (Type: location) - Main corporate office building
Entity 4: hr policies document (Type: document) - Employee handbook and HR policies
Entity 5: marketing team (Type: team) - Marketing department team
Entity 6: product launch (Type: event) - New product announcement and launch event
Entity 7: training program (Type: project) - Employee development training initiative

Output:
("new_node"{tuple_delimiter}"Corporate Operations Hub"{tuple_delimiter}"team"{tuple_delimiter}"Centralized operations team managing office facilities, document management, and administrative processes"{tuple_delimiter}0.8){record_delimiter}
("new_node"{tuple_delimiter}"Business Events & Training Center"{tuple_delimiter}"project"{tuple_delimiter}"Event coordination and training project managing corporate meetings, product launches, and employee development"{tuple_delimiter}0.9){record_delimiter}
("new_edge"{tuple_delimiter}"conference room booking"{tuple_delimiter}"Corporate Operations Hub"{tuple_delimiter}"Administrative task managed by corporate operations for facility management"{tuple_delimiter}"facility management, administrative operations"{tuple_delimiter}8){record_delimiter}
("new_edge"{tuple_delimiter}"office headquarters"{tuple_delimiter}"Corporate Operations Hub"{tuple_delimiter}"Physical location managed and maintained by corporate operations"{tuple_delimiter}"facility management, location operations"{tuple_delimiter}8){record_delimiter}
("new_edge"{tuple_delimiter}"hr policies document"{tuple_delimiter}"Corporate Operations Hub"{tuple_delimiter}"Corporate documentation managed by administrative operations"{tuple_delimiter}"document management, corporate policies"{tuple_delimiter}7){record_delimiter}
("new_edge"{tuple_delimiter}"quarterly meeting"{tuple_delimiter}"Business Events & Training Center"{tuple_delimiter}"Regular corporate event coordinated by business events project"{tuple_delimiter}"event coordination, business meetings"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"product launch"{tuple_delimiter}"Business Events & Training Center"{tuple_delimiter}"Major business event managed by events and training coordination"{tuple_delimiter}"event coordination, product management"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"training program"{tuple_delimiter}"Business Events & Training Center"{tuple_delimiter}"Employee development project that fits directly into training center operations"{tuple_delimiter}"training coordination, employee development"{tuple_delimiter}9){record_delimiter}
("new_edge"{tuple_delimiter}"marketing team"{tuple_delimiter}"Corporate Management"{tuple_delimiter}"Cross-functional team that works with multiple operational areas - direct parent connection"{tuple_delimiter}"cross-functional, team coordination"{tuple_delimiter}6){record_delimiter}
("major_category"{tuple_delimiter}"Business Events & Training Center"{tuple_delimiter}"Strategic business coordination that drives company growth and employee development"){completion_delimiter}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must match the user's question language.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, cite a maximum of 5 most relevant sources used.
- Use the following formats for citations: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""
