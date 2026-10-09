"""
AmaniBench v1.0 - Evaluation Benchmark for Kenyan Civic RAG Systems
====================================================================

A comprehensive 100-question evaluation dataset specifically designed for
assessing Kenyan civic AI systems across 5 critical categories.

Categories (20 questions each):
1. Sheng Understanding - Tests ability to parse Kenyan street slang
2. Legal Citation Accuracy - Tests precision in legal references
3. Daily Life Impact - Tests practical explanation of policies
4. Ambiguity Handling - Tests understanding of Kenyan colloquialisms
5. Temporal Awareness - Tests understanding of recent events timeline

Usage:
    python amanibench_generator.py > amanibench_v1.jsonl

Author: AmaniQuery Team
Date: November 2025
"""

import json
from typing import List, Dict, Any


def generate_amanibench() -> List[Dict[str, Any]]:
    """Generate all 100 benchmark questions"""

    benchmark = []

    # ========================================================================
    # CATEGORY 1: SHENG UNDERSTANDING (20 questions)
    # ========================================================================

    sheng_questions = [
        {
            "category": "sheng_understanding",
            "query": "Kanjo wameamua nini kuhusu parking doh?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Nairobi City County",
                "parking fees",
                "KES 300",
                "CBD",
                "increase",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "sheng_understanding",
            "query": "Bunge wanapanga kuongeza tax ya mat ama?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Parliament",
                "public service vehicles",
                "matatu",
                "taxation",
                "proposal",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "sheng_understanding",
            "query": "Hii Finance Bill inasema nini kuhusu maji na stima?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Finance Bill",
                "water services",
                "electricity",
                "VAT",
                "utilities",
            ],
            "acceptable_languages": ["mixed", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Mheshimiwa wa Starehe alisema aje kuhusu housing levy?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Member of Parliament",
                "Starehe constituency",
                "housing levy",
                "3%",
                "opinion",
            ],
            "acceptable_languages": ["mixed", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Serikali wanataka kupunguza doh ya healthcare ama waongeze?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "government",
                "healthcare budget",
                "increase or decrease",
                "allocation",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini maana ya 'Kidero grass' na ilitumika wapi?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kidero grass",
                "Nairobi beautification",
                "Governor Kidero",
                "city aesthetics",
                "cost controversy",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "hard",
        },
        {
            "category": "sheng_understanding",
            "query": "Gava ilisema nini kuhusu bei ya unga?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "government",
                "maize flour price",
                "subsidy",
                "food security",
                "cost of living",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "sheng_understanding",
            "query": "Hii Hustler Fund inasaidia aje wasee wa biashara ndogo?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Hustler Fund",
                "small businesses",
                "loans",
                "financial inclusion",
                "government initiative",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini shida na barabara ya Eastern Bypass, na gava inafanya nini?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Eastern Bypass",
                "traffic congestion",
                "road expansion",
                "infrastructure projects",
                "government plans",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Waziri wa Elimu alisema nini kuhusu CBC na exams?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Education Cabinet Secretary",
                "CBC curriculum",
                "examinations",
                "education reforms",
                "junior secondary",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Kuna sheria mpya kuhusu kutupa takataka ovyo ovyo Nairobi?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Nairobi County",
                "waste management",
                "environmental laws",
                "fines",
                "public sanitation",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini maana ya 'Azimio' na 'Kenya Kwanza' kwa siasa za Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Azimio la Umoja",
                "Kenya Kwanza",
                "political coalitions",
                "general elections",
                "political ideologies",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "sheng_understanding",
            "query": "Bei ya mafuta imepanda ama imeshuka wiki hii?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "fuel prices",
                "EPRA",
                "petrol",
                "diesel",
                "price review",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini mpango wa serikali kuhusu vijana wasio na kazi?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "youth unemployment",
                "government programs",
                "job creation",
                "internships",
                "skills training",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Kuna sheria gani mpya kuhusu matumizi ya plastic bags?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "plastic bags ban",
                "environmental protection",
                "NEMA",
                "alternative packaging",
                "fines",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini maana ya 'BBI' na ilikuwa inataka kufanya nini?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Building Bridges Initiative",
                "constitutional amendments",
                "political reforms",
                "national unity",
                "referendum",
            ],
            "acceptable_languages": ["sheng", "mixed", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "sheng_understanding",
            "query": "Serikali inafanya nini kusaidia wakulima wa kahawa?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "coffee farmers",
                "subsidies",
                "market access",
                "agricultural reforms",
                "government support",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini mpango wa Nairobi County kuhusu mabasi ya BRT?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Nairobi County",
                "BRT system",
                "public transport",
                "traffic decongestion",
                "infrastructure",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "sheng_understanding",
            "query": "Kuna sheria mpya kuhusu data privacy na simu zetu?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "data protection act",
                "privacy laws",
                "mobile data",
                "digital rights",
                "CAK",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "hard",
        },
        {
            "category": "sheng_understanding",
            "query": "Nini maana ya 'mzinga' kwa lugha ya mtaani na inahusiana na nini?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "mzinga",
                "slang",
                "alcohol",
                "illicit brews",
                "public health concern",
            ],
            "acceptable_languages": ["sheng", "mixed"],
            "difficulty": "hard",
        },
    ]
    benchmark.extend(sheng_questions)

    # ========================================================================
    # CATEGORY 2: LEGAL CITATION ACCURACY (20 questions)
    # ========================================================================

    legal_questions = [
        {
            "category": "legal_citation",
            "query": "What are the fundamental rights and freedoms guaranteed under Article 25 of the Constitution of Kenya?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Article 25",
                "Fundamental rights and freedoms",
                "non-derogable rights",
                "freedom from torture",
                "freedom from slavery",
                "right to a fair trial",
                "right to an order of habeas corpus",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Cite the section of the Employment Act that deals with unfair termination of employment.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Employment Act",
                "unfair termination",
                "Section 45",
                "grounds for termination",
                "procedure for termination",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "According to the Land Act, what are the different categories of land ownership in Kenya?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Land Act",
                "categories of land",
                "public land",
                "community land",
                "private land",
                "land tenure",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
        {
            "category": "legal_citation",
            "query": "What does Section 4 of the Data Protection Act state regarding principles of data protection?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Data Protection Act",
                "principles of data protection",
                "lawfulness",
                "fairness",
                "transparency",
                "purpose limitation",
                "data minimization",
                "accuracy",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "legal_citation",
            "query": "Outline the main functions of the National Environmental Management Authority (NEMA) as per the EMCA.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "NEMA",
                "Environmental Management and Co-ordination Act (EMCA)",
                "functions of NEMA",
                "environmental policy",
                "coordination",
                "enforcement",
                "environmental impact assessments",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Provide the legal definition of 'hearsay evidence' under the Evidence Act.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Evidence Act",
                "hearsay evidence",
                "definition",
                "admissibility",
                "exceptions to hearsay rule",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "legal_citation",
            "query": "What are the essential elements of the offence of 'theft' as defined in the Penal Code?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Penal Code",
                "theft",
                "definition of theft",
                "fraudulent intent",
                "taking property without consent",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Describe the requirements for a valid will in Kenya according to the Law of Succession Act.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Law of Succession Act",
                "valid will",
                "testator's capacity",
                "writing requirement",
                "signature",
                "attestation by witnesses",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "What does Article 232 of the Constitution state about the values and principles of public service?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Article 232",
                "public service",
                "values and principles",
                "high standards of professional ethics",
                "efficient, effective and economic use of resources",
                "accountability",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
        {
            "category": "legal_citation",
            "query": "Summarize the provisions for bail and bond in the Criminal Procedure Code.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Criminal Procedure Code",
                "bail and bond",
                "right to bail",
                "conditions for bail",
                "factors considered by court",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Under the Children Act, what constitutes 'parental responsibility'?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Children Act",
                "parental responsibility",
                "definition",
                "duty to maintain",
                "duty to protect",
                "duty to educate",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
        {
            "category": "legal_citation",
            "query": "What are the various forms of intellectual property recognized under Kenyan law, and which statutes govern them?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "intellectual property",
                "copyright",
                "trademarks",
                "patents",
                "industrial designs",
                "Copyright Act",
                "Industrial Property Act",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "legal_citation",
            "query": "Explain the concept of 'judicial review' in Kenya, citing relevant constitutional provisions.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "judicial review",
                "Constitution of Kenya",
                "Article 23",
                "High Court's jurisdiction",
                "legality of administrative action",
                "prerogative orders (certiorari, mandamus, prohibition)",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "legal_citation",
            "query": "According to the Public Procurement and Asset Disposal Act, what are the core principles guiding public procurement?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Public Procurement and Asset Disposal Act",
                "procurement principles",
                "fairness",
                "transparency",
                "accountability",
                "cost-effectiveness",
                "competition",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "What are the legal implications of 'contempt of court' as per the Contempt of Court Act?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Contempt of Court Act",
                "contempt of court",
                "civil contempt",
                "criminal contempt",
                "penalties",
                "disobedience of court orders",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Describe the process for amending the Constitution of Kenya as provided for in Chapter 16.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Constitution of Kenya",
                "amendment process",
                "Chapter 16",
                "parliamentary initiative",
                "popular initiative",
                "referendum",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "legal_citation",
            "query": "What does the Consumer Protection Act stipulate regarding product liability and consumer rights?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Consumer Protection Act",
                "product liability",
                "consumer rights",
                "right to safety",
                "right to information",
                "right to redress",
                "defective products",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "legal_citation",
            "query": "Under the Traffic Act, what are the penalties for driving under the influence of alcohol?",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Traffic Act",
                "driving under influence (DUI)",
                "blood alcohol limit",
                "penalties",
                "fines",
                "imprisonment",
                "suspension of license",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
    ]
    benchmark.extend(legal_questions)

    # ========================================================================
    # CATEGORY 3: DAILY LIFE IMPACT (20 questions)
    # ========================================================================

    impact_questions = [
        {
            "category": "daily_impact",
            "query": "How does inflation affect the cost of basic food items like maize flour and cooking oil in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kenya National Bureau of Statistics (KNBS)",
                "inflation rate",
                "Consumer Price Index (CPI)",
                "cost of living",
                "food basket prices",
                "government subsidies",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "daily_impact",
            "query": "What are the common types of taxes individuals pay in Kenya, apart from income tax?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Value Added Tax (VAT)",
                "Excise Duty",
                "Pay As You Earn (PAYE)",
                "Housing Levy",
                "National Social Security Fund (NSSF)",
                "National Hospital Insurance Fund (NHIF)",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "daily_impact",
            "query": "Can I get a refund for a faulty product purchased from a supermarket in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Consumer Protection Act",
                "right to goods of merchantable quality",
                "right to refund/replacement",
                "proof of purchase (receipt)",
                "return policy",
                "Kenya Bureau of Standards (KEBS)",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "daily_impact",
            "query": "What are the typical charges for withdrawing money from an ATM of a different bank?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "interbank ATM transaction fees",
                "bank charges",
                "transaction limits",
                "using own bank's ATM",
                "mobile banking alternatives",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
        {
            "category": "daily_impact",
            "query": "How does the ban on single-use plastics affect small businesses and consumers in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "National Environment Management Authority (NEMA)",
                "plastic bag ban (2017)",
                "single-use plastic ban (2020)",
                "alternative packaging materials",
                "cost implications for businesses",
                "environmental benefits",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "daily_impact",
            "query": "What are the main requirements for someone to volunteer for community service in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "NGOs and CBOs",
                "area of interest (e.g., environment, education, health)",
                "age requirements",
                "skills match",
                "background checks (for vulnerable groups)",
                "local community initiatives",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "daily_impact",
            "query": "How can I report a power outage or a faulty electricity pole to KPLC?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "KPLC customer care",
                "USSD code (*977#)",
                "MyPower app",
                "social media channels",
                "nearest KPLC office",
                "account number/meter number",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "daily_impact",
            "query": "What are the legal steps involved in buying a piece of land in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "land search (Ministry of Lands)",
                "mutation forms/survey plans",
                "site visit",
                "valuation",
                "sale agreement",
                "transfer of ownership",
                "stamp duty",
                "land rates clearance",
                "lawyer engagement",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
    ]
    benchmark.extend(impact_questions)

    # ========================================================================
    # CATEGORY 4: AMBIGUITY HANDLING (20 questions)
    # ========================================================================

    ambiguity_questions = [
        {
            "category": "ambiguity",
            "query": "What is happening at The Hill?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Parliament of Kenya",
                "legislative activities",
                "parliamentary debates",
                "bills under consideration",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "State House announced new measures today",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Office of the President",
                "Presidential directives",
                "government policy",
                "national announcements",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "What's the latest on the budget?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "National Budget",
                "County Budget",
                "Fiscal Year 2024/2025",
                "Supplementary Budget",
                "Treasury Cabinet Secretary",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "Is the market going up or down?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Nairobi Securities Exchange (NSE)",
                "stock market performance",
                "commodity prices (e.g., fuel, maize)",
                "economic indicators",
                "inflation",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "Kenyans are talking about the new law.",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Finance Act 2024",
                "Housing Act",
                "Digital Economy Bill",
                "Public Participation",
                "Parliamentary proceedings",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "What's the situation at the border?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kenya-Somalia border",
                "Kenya-Uganda border",
                "Kenya-Tanzania border",
                "security operations",
                "trade activities",
                "refugee movements",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "ambiguity",
            "query": "The court made a ruling today.",
            "expected_query_type": "legal",
            "golden_answer_facts": [
                "Supreme Court",
                "Court of Appeal",
                "High Court",
                "Magistrate Court",
                "specific case name/topic",
                "judicial pronouncement",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "Is M-Pesa working?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Safaricom network status",
                "system downtime",
                "transaction issues",
                "M-Pesa services (send money, pay bill, etc.)",
                "customer care",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "What are the new rules for visitors?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "immigration policies",
                "visa requirements",
                "eTA (Electronic Travel Authorization)",
                "COVID-19 travel protocols (if applicable)",
                "customs regulations",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "The President is on a tour.",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "President William Ruto",
                "domestic tour (e.g., specific county)",
                "international visit (e.g., specific country)",
                "purpose of visit (e.g., development projects, bilateral talks)",
                "presidential engagements",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "What's the price of unga?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "maize flour",
                "retail price",
                "brand variations (e.g., sifted, unsifted)",
                "regional price differences",
                "government subsidies",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "Did the bill pass?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "specific bill name (e.g., Finance Bill, Affordable Housing Bill)",
                "Parliament (National Assembly, Senate)",
                "assent by President",
                "gazettement",
                "legislative process",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "What's the outlook for the economy?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "economic growth forecast (GDP)",
                "inflation rate",
                "exchange rate (KES to USD)",
                "interest rates",
                "employment outlook",
                "Central Bank of Kenya (CBK) reports",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "hard",
        },
        {
            "category": "ambiguity",
            "query": "The school year starts soon.",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Ministry of Education calendar",
                "primary school",
                "secondary school",
                "university/college",
                "specific term (e.g., Term 1, Term 2)",
                "academic year",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "They are building a new road.",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kenya National Highways Authority (KENHA)",
                "Kenya Rural Roads Authority (KERRA)",
                "specific road project (e.g., Nairobi Expressway, bypasses)",
                "location/region",
                "infrastructure development",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "What's the latest on the SGR?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Standard Gauge Railway (SGR)",
                "passenger services",
                "cargo services",
                "extension plans (e.g., Naivasha, Kisumu)",
                "debt repayment",
                "operations and management",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "Is there a new housing project?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Affordable Housing Program",
                "government initiatives",
                "private sector developments",
                "location (e.g., specific estate, county)",
                "eligibility criteria",
            ],
            "acceptable_languages": ["english"],
            "difficulty": "medium",
        },
        {
            "category": "ambiguity",
            "query": "What's the weather like in the Highlands?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kenya Meteorological Department",
                "Central Highlands (e.g., Nyeri, Murang'a)",
                "Rift Valley Highlands (e.g., Nakuru, Kericho)",
                "rainfall patterns",
                "temperatures",
                "forecast for specific days",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "The police are looking for suspects.",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Kenya Police Service",
                "Directorate of Criminal Investigations (DCI)",
                "specific crime/incident (e.g., robbery, murder)",
                "public appeal for information",
                "ongoing investigations",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "ambiguity",
            "query": "Who won the match?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "specific sports (e.g., football, rugby, athletics)",
                "teams/individuals involved",
                "league/tournament (e.g., FKF Premier League, Kenya Sevens)",
                "recent results",
                "upcoming fixtures",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "easy",
        },
    ]
    benchmark.extend(ambiguity_questions)

    # ========================================================================
    # CATEGORY 5: TEMPORAL AWARENESS (20 questions)
    # ========================================================================

    temporal_questions = [
        {
            "category": "temporal",
            "query": "What happened to Finance Bill 2024 after June protests?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "withdrawn by President",
                "June 26 2024",
                "mass protests",
                "Bill rejected",
                "not enacted",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "easy",
        },
        {
            "category": "temporal",
            "query": "Is Housing Levy still being collected in 2025?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "yes, employee 3%",
                "employer contribution suspended",
                "court cases pending",
                "Finance Act 2023 provisions",
            ],
            "acceptable_languages": ["english", "swahili", "mixed"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "When was the last general election held in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "August 9, 2022",
                "William Ruto elected President",
                "IEBC",
                "presidential, parliamentary, county elections",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "temporal",
            "query": "What is the current status of the SGR project in Kenya as of 2024?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "operational from Mombasa to Naivasha",
                "extension to Kisumu/Malaba stalled",
                "debt repayment to China Exim Bank",
                "freight and passenger services",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "Has the Competency-Based Curriculum (CBC) been fully implemented across all grades by 2025?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "phased implementation",
                "Junior Secondary School (JSS) challenges",
                "Presidential Working Party on Education Reforms (PWPER)",
                "Grade 8 transition",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "What were the key economic indicators for Kenya in the last quarter of 2023?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "GDP growth rate",
                "inflation rate",
                "exchange rate (KES to USD)",
                "interest rates (CBR)",
                "exports/imports",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "When is the next national census scheduled to take place in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "every 10 years",
                "next in 2029",
                "Kenya National Bureau of Statistics (KNBS)",
                "population, housing, demographic data",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "temporal",
            "query": "What major sporting events are scheduled for Kenya in 2025?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "World Athletics Championships (if applicable)",
                "Safari Rally WRC",
                "local football leagues (FKF Premier League)",
                "rugby tournaments (Kenya Sevens)",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "Has the government achieved its target of planting 15 billion trees by 2032 as of 2024?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "ongoing initiative",
                "progress reports",
                "National Tree Growing Day",
                "Ministry of Environment",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "What is the history of devolution in Kenya since the 2010 constitution?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "county governments established 2013",
                "Council of Governors (CoG)",
                "functions transferred from national government",
                "challenges and successes",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "When did Kenya gain independence?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "December 12, 1963",
                "Jomo Kenyatta first President",
                "from British colonial rule",
                "Jamhuri Day",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "temporal",
            "query": "What was the impact of the 2007 post-election violence on Kenya's political landscape in the years that followed?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Grand Coalition Government",
                "new constitution 2010",
                "ICC cases",
                "electoral reforms",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "Are there any new regulations regarding digital lenders expected in Kenya in late 2024?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Central Bank of Kenya (CBK) licensing",
                "Digital Credit Providers (DCPs) Act",
                "consumer protection",
                "interest rate caps/transparency",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "What was the average rainfall in Nairobi during the long rains season of 2023?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "March to May",
                "Kenya Meteorological Department data",
                "impact on agriculture/water levels",
                "comparison to historical averages",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "When was the last time Kenya hosted a major international conference?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Africa Climate Summit (ACS) 2023",
                "UNEP headquarters in Nairobi",
                "TICAD, WTO Ministerial Conference",
                "venue: KICC",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "What is the history of the Kenyan shilling's performance against the US dollar over the last five years?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "depreciation trends",
                "Central Bank interventions",
                "factors: imports, debt, remittances",
                "exchange rate data",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "Has the universal health coverage (UHC) program been fully rolled out across all counties by 2024?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "phased implementation",
                "NHIF reforms",
                "Community Health Promoters (CHPs)",
                "challenges in funding/infrastructure",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "medium",
        },
        {
            "category": "temporal",
            "query": "When was the first female Chief Justice appointed in Kenya?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "Martha Koome",
                "May 2021",
                "Judicial Service Commission (JSC)",
                "Supreme Court",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "easy",
        },
        {
            "category": "temporal",
            "query": "What is the timeline for the completion of the Lamu Port-South Sudan-Ethiopia Transport (LAPSSET) Corridor project?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "ongoing multi-national project",
                "phased development (port, roads, railway, pipeline)",
                "initial berths operational",
                "long-term vision",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
        {
            "category": "temporal",
            "query": "What were the major changes to the Kenyan education system in the last decade?",
            "expected_query_type": "public_interest",
            "golden_answer_facts": [
                "transition from 8-4-4 to CBC",
                "free primary and day secondary education",
                "TVET reforms",
                "university funding model",
            ],
            "acceptable_languages": ["english", "swahili"],
            "difficulty": "hard",
        },
    ]
    benchmark.extend(temporal_questions)

    # Combine all questions
    benchmark.extend(sheng_questions)
    benchmark.extend(legal_questions)
    benchmark.extend(impact_questions)
    benchmark.extend(ambiguity_questions)
    benchmark.extend(temporal_questions)

    return benchmark


def print_ndjson(benchmark: List[Dict]):
    """Print benchmark as NDJSON"""
    for item in benchmark:
        print(json.dumps(item, ensure_ascii=False))


def save_ndjson(benchmark: List[Dict], filename: str = "amanibench_v1.jsonl"):
    """Save benchmark to NDJSON file"""
    with open(filename, "w", encoding="utf-8") as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(benchmark)} questions to {filename}")


def print_stats(benchmark: List[Dict]):
    """Print benchmark statistics"""
    from collections import Counter

    print("\n" + "=" * 80)
    print("AMANIBENCH v1.0 STATISTICS")
    print("=" * 80)

    print(f"\nTotal Questions: {len(benchmark)}")

    # By category
    categories = Counter(q["category"] for q in benchmark)
    print("\nBy Category:")
    for cat, count in categories.items():
        print(f"  {cat:25s}: {count:3d} questions")

    # By difficulty
    difficulties = Counter(q["difficulty"] for q in benchmark)
    print("\nBy Difficulty:")
    for diff, count in difficulties.items():
        print(f"  {diff:10s}: {count:3d} questions")

    # By expected query type
    types = Counter(q["expected_query_type"] for q in benchmark)
    print("\nBy Expected Query Type:")
    for qtype, count in types.items():
        print(f"  {qtype:20s}: {count:3d} questions")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Generating AmaniBench v1.0...")

    benchmark = generate_amanibench()

    # Print statistics
    print_stats(benchmark)

    # Save to file
    save_ndjson(benchmark, "amanibench_v1.jsonl")

    # Print first 3 examples
    print("\n" + "=" * 80)
    print("SAMPLE QUESTIONS:")
    print("=" * 80)

    for i, q in enumerate(benchmark[:3], 1):
        print(f"\n{i}. [{q['category']}] {q['query']}")
        print(f"   Type: {q['expected_query_type']}")
        print(f"   Facts: {', '.join(q['golden_answer_facts'][:3])}...")
        print(f"   Difficulty: {q['difficulty']}")

    print("\n" + "=" * 80)
    print("USAGE:")
    print("=" * 80)
    print("""
# Evaluate your RAG system:
python evaluate_with_amanibench.py \\
    --benchmark amanibench_v1.jsonl \\
    --model your_model_name \\
    --output results.json

# Filter by category:
cat amanibench_v1.jsonl | grep '"category":"sheng_understanding"' > sheng_only.jsonl

# Get statistics:
python -c "from amanibench_generator import *; print_stats(generate_amanibench())"
    """)
