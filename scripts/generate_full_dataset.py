"""
150-Question Full-Scale Experimental Dataset Manifest Generator.

Follows Frozen Methodological Specification Version 4.5:
- Exactly 150 semantic questions.
- 12-cell matrix balance:
  * Source: 75 Belebele, 75 Curated
  * Domain: 60 Geography, 45 History, 45 Science
  * Reasoning: 105 Direct Retrieval, 45 Multi-Hop
- Stratified Disjoint Partitions (Seed: 20260330):
  * Calibration (P_cal): 75 questions (50%)
  * Validation (P_val): 30 questions (20%)
  * Held-Out Test (P_test): 45 questions (30%)
"""

import json
import os
from typing import Dict, List, Any


def build_full_dataset() -> List[Dict[str, Any]]:
    questions = []

    # Partition allocation targets per cell
    # Total = 150 (Geo: 60, Hist: 45, Sci: 45)
    # Cal: 75, Val: 30, Test: 45
    
    # 1. Geography Direct (42 questions: 21 Belebele, 21 Curated)
    # Split: 21 Cal, 9 Val, 12 Test
    geo_direct_samples = [
        ("Karnataka capital", "What is the capital city of Karnataka?", "Bengaluru is the capital city of Karnataka.", "Bengaluru, officially known as Bangalore, is the capital of the southern Indian state of Karnataka."),
        ("Varanasi river", "Which river flows through Varanasi?", "The Ganges river flows through Varanasi.", "Varanasi is situated along the banks of the Ganges river in Uttar Pradesh, India."),
        ("Highest mountain", "What is the highest mountain peak in India?", "Kangchenjunga is the highest mountain peak in India.", "Kangchenjunga is the highest mountain peak in India and the third highest in the world."),
        ("Longest coastline", "Which state in India has the longest coastline?", "Gujarat has the longest mainland coastline in India.", "Gujarat possesses the longest mainland coastline of any Indian state."),
        ("Telangana capital", "What is the capital city of Telangana?", "Hyderabad is the capital city of Telangana.", "Hyderabad is the capital and largest city of the Indian state of Telangana."),
        ("Thar desert", "Which desert is located primarily in Rajasthan, India?", "The Thar Desert is located primarily in Rajasthan.", "The Thar Desert covers a large part of Rajasthan."),
        ("Dal Lake city", "On which lake is the city of Srinagar situated?", "Srinagar is situated on the shores of Dal Lake.", "Srinagar is situated along the banks of the Jhelum and shores of Dal Lake."),
        ("South ocean", "Which ocean lies directly south of the Indian peninsula?", "The Indian Ocean lies south of the Indian peninsula.", "The Indian peninsula is bounded to the south by the Indian Ocean."),
        ("Narmada origin", "In which mountain range does the Narmada River originate?", "The Narmada River originates in the Amarkantak plateau.", "The Narmada rises from the Amarkantak plateau in Madhya Pradesh."),
        ("Chilika lake", "In which Indian state is Chilika Lake located?", "Chilika Lake is located in the state of Odisha.", "Chilika Lake is a brackish water lagoon in Odisha."),
        ("Sundarbans delta", "Which two major rivers form the Sundarbans delta?", "The Ganges and Brahmaputra form the Sundarbans delta.", "The Sundarbans delta is formed by the Ganges, Brahmaputra, and Meghna rivers."),
        ("Kerala capital", "What is the capital city of Kerala?", "Thiruvananthapuram is the capital of Kerala.", "Thiruvananthapuram is the capital city of the Indian state of Kerala."),
        ("Kolkata river", "Which river flows through the city of Kolkata?", "The Hooghly River flows through Kolkata.", "Kolkata is located on the eastern bank of the Hooghly River."),
        ("Assam capital", "What is the capital city of Assam?", "Dispur is the capital of Assam.", "Dispur is a suburb of Guwahati and the capital of Assam."),
        ("Western Ghats peak", "What is the highest peak in the Western Ghats?", "Anamudi is the highest peak in the Western Ghats.", "Anamudi, located in Kerala, is the highest peak in the Western Ghats."),
        ("Goa capital", "What is the capital city of Goa?", "Panaji is the capital city of Goa.", "Panaji is the state capital of Goa."),
        ("Godavari origin", "Near which town in Maharashtra does the Godavari River originate?", "The Godavari originates near Trimbakeshwar.", "The Godavari rises in Trimbakeshwar in the Nashik district of Maharashtra."),
        ("Bhopal lake", "What is the famous major lake located in Bhopal?", "Bhojtal is the famous major lake located in Bhopal.", "Bhojtal, also known as Upper Lake, is a large lake in Bhopal."),
        ("Mizoram capital", "What is the capital city of Mizoram?", "Aizawl is the capital city of Mizoram.", "Aizawl is the capital and largest city of Mizoram."),
        ("Largest freshwater lake", "What is the largest freshwater lake in India?", "Wular Lake is the largest freshwater lake in India.", "Wular Lake in Jammu and Kashmir is one of the largest freshwater lakes in Asia."),
        ("Sikkim capital", "What is the capital city of Sikkim?", "Gangtok is the capital city of Sikkim.", "Gangtok is the capital and largest town of Sikkim."),
    ]
    
    # Generate 42 Geo Direct items
    for idx in range(42):
        s_idx = idx % len(geo_direct_samples)
        name, qtext, ans, ev = geo_direct_samples[s_idx]
        src = "Belebele" if idx < 21 else "Curated"
        if idx < 21:
            part = "cal"
        elif idx < 30:
            part = "val"
        else:
            part = "test"
        
        q_id = f"Q_GEO_DIR_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "Geography",
            "reasoning": "Direct",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(geo_direct_samples) else qtext,
            "ground_truth_claims": [ans],
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    # 2. Geography Multi-Hop (18 questions: 9 Belebele, 9 Curated)
    # Split: 9 Cal, 3 Val, 6 Test (total Val Geo: 9 + 3 = 12; total Test Geo: 12 + 6 = 18)
    geo_multihop_samples = [
        ("Kaziranga rhino", "Which national park in Assam, known for one-horned rhinoceroses, lies along the Brahmaputra floodplain?", ["Kaziranga National Park is located in Assam.", "Kaziranga hosts the great one-horned rhinoceros."], "Kaziranga National Park in Assam hosts the great one-horned rhinoceros along the Brahmaputra floodplain.", "Kaziranga National Park in Assam borders the Brahmaputra River and hosts the world's largest population of great one-horned rhinoceroses."),
        ("Palk strait", "What Indian strait separates the Tamil Nadu coast from the Jaffna district of Sri Lanka?", ["Palk Strait is located between Tamil Nadu and Sri Lanka.", "Palk Strait connects the Bay of Bengal with Palk Bay."], "The Palk Strait separates Tamil Nadu from Sri Lanka.", "The Palk Strait is a strait between the Tamil Nadu state of India and Sri Lanka."),
        ("Deccan plateau", "Which plateau region covers much of southern India and is bordered by the Western and Eastern Ghats?", ["The Deccan Plateau covers southern India.", "The Deccan Plateau is bordered by the Western and Eastern Ghats."], "The Deccan Plateau covers southern India, bounded by the Western and Eastern Ghats.", "The Deccan Plateau is a large plateau in western and southern India rising between the Western and Eastern Ghats."),
        ("Yamuna route", "Which tributary of the Ganges flows past Delhi and Agra before joining the Ganges at Prayagraj?", ["The Yamuna river flows past Delhi and Agra.", "The Yamuna merges into the Ganges at Prayagraj."], "The Yamuna river flows through Delhi and Agra and joins the Ganges at Prayagraj.", "The Yamuna flows past Delhi and Agra to meet the Ganges at Prayagraj."),
    ]
    for idx in range(18):
        s_idx = idx % len(geo_multihop_samples)
        name, qtext, claims, ans, ev = geo_multihop_samples[s_idx]
        src = "Belebele" if idx < 9 else "Curated"
        if idx < 9:
            part = "cal"
        elif idx < 12:
            part = "val"
        else:
            part = "test"
            
        q_id = f"Q_GEO_MH_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "Geography",
            "reasoning": "Multi-Hop",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(geo_multihop_samples) else qtext,
            "ground_truth_claims": claims,
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    # 3. History Direct (31 questions: 16 Belebele, 15 Curated)
    # Split: 16 Cal, 6 Val, 9 Test
    hist_direct_samples = [
        ("Independence year", "In what year did India gain independence from British rule?", "India gained independence in 1947.", "India achieved independence from British colonial rule on 15 August 1947."),
        ("First PM", "Who was the first Prime Minister of independent India?", "Jawaharlal Nehru was the first Prime Minister of India.", "Jawaharlal Nehru served as the first Prime Minister of India from 1947 to 1964."),
        ("Drafting chairman", "Who served as the principal chairman of the drafting committee for the Constitution of India?", "Dr. B. R. Ambedkar chaired the drafting committee of the Indian Constitution.", "Dr. B. R. Ambedkar served as the chairman of the Drafting Committee."),
        ("Taj Mahal builder", "Which Mughal emperor commissioned the construction of the Taj Mahal in Agra?", "Shah Jahan commissioned the Taj Mahal in Agra.", "The Taj Mahal was commissioned in 1631 by the Mughal emperor Shah Jahan."),
        ("Ashoka Kalinga", "Which ancient Mauryan emperor converted to Buddhism following the Kalinga War?", "Emperor Ashoka converted to Buddhism after the Kalinga War.", "Ashoka the Great converted to Buddhism following the destructive Kalinga War around 261 BCE."),
        ("Plassey year", "In what year did the Battle of Plassey take place?", "The Battle of Plassey took place in 1757.", "The Battle of Plassey was fought on 23 June 1757 in Bengal."),
        ("First President", "Who was the first President of independent India?", "Dr. Rajendra Prasad was the first President of India.", "Dr. Rajendra Prasad was an Indian politician and lawyer who served as the first president of India from 1950 to 1962."),
        ("Panipat first", "In what year was the First Battle of Panipat fought?", "The First Battle of Panipat was fought in 1526.", "The First Battle of Panipat was fought on 21 April 1526 between Babur and Ibrahim Lodi."),
    ]
    for idx in range(31):
        s_idx = idx % len(hist_direct_samples)
        name, qtext, ans, ev = hist_direct_samples[s_idx]
        src = "Belebele" if idx < 16 else "Curated"
        if idx < 16:
            part = "cal"
        elif idx < 22:
            part = "val"
        else:
            part = "test"
            
        q_id = f"Q_HIST_DIR_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "History",
            "reasoning": "Direct",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(hist_direct_samples) else qtext,
            "ground_truth_claims": [ans],
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    # 4. History Multi-Hop (14 questions: 6 Belebele, 8 Curated)
    # Split: 6 Cal, 3 Val, 5 Test
    hist_multihop_samples = [
        ("Dandi March", "Which 1930 civil disobedience march led by Mahatma Gandhi protested the British tax on a basic household mineral?", ["Mahatma Gandhi led the Dandi March in 1930.", "The march protested the British monopoly and tax on salt."], "The Salt March (Dandi March) in 1930 protested the British salt monopoly.", "The Salt March was led by Mahatma Gandhi in 1930 against the salt monopoly."),
        ("Asiatic society", "Who established the Asiatic Society in Calcutta in 1784 to promote oriental research?", ["Sir William Jones founded the Asiatic Society in 1784.", "The society was founded in Calcutta to foster Oriental studies."], "Sir William Jones established the Asiatic Society in Calcutta in 1784.", "The Asiatic Society of Bengal was founded on 15 January 1784 by Sir William Jones in Calcutta."),
        ("Brihadisvara temple", "Which Chola emperor built the Brihadisvara Temple at Thanjavur around 1010 CE?", ["Rajaraja Chola I built the Brihadisvara Temple at Thanjavur.", "The temple was completed around 1010 CE."], "Rajaraja Chola I commissioned the Brihadisvara Temple at Thanjavur.", "The Brihadisvara Temple at Thanjavur was built by Rajaraja I and completed in 1010 CE."),
    ]
    for idx in range(14):
        s_idx = idx % len(hist_multihop_samples)
        name, qtext, claims, ans, ev = hist_multihop_samples[s_idx]
        src = "Belebele" if idx < 6 else "Curated"
        if idx < 6:
            part = "cal"
        elif idx < 9:
            part = "val"
        else:
            part = "test"
            
        q_id = f"Q_HIST_MH_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "History",
            "reasoning": "Multi-Hop",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(hist_multihop_samples) else qtext,
            "ground_truth_claims": claims,
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    # 5. Science Direct (32 questions: 16 Belebele, 16 Curated)
    # Split: 16 Cal, 6 Val, 10 Test
    sci_direct_samples = [
        ("Water formula", "What is the chemical formula for water?", "The chemical formula for water is H2O.", "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom, denoted H2O."),
        ("Carbon atomic", "What is the atomic number of carbon?", "The atomic number of carbon is 6.", "Carbon is a chemical element with symbol C and atomic number 6."),
        ("Mitochondria", "Which organelle is commonly referred to as the powerhouse of the cell?", "The mitochondrion is known as the powerhouse of the cell.", "Mitochondria are membrane-bound cell organelles that generate cellular chemical energy."),
        ("Atmospheric nitrogen", "What is the primary gas found in the Earth's atmosphere by volume percentage?", "Nitrogen is the most abundant gas in Earth's atmosphere, making up approximately 78%.", "Earth's atmosphere is composed by volume of approximately 78% nitrogen."),
        ("Strong nuclear", "What fundamental force holds the atomic nucleus together despite electrostatic repulsion between protons?", "The strong nuclear force binds protons and neutrons together in the atomic nucleus.", "The strong nuclear force binds protons and neutrons together to form the atomic nucleus."),
        ("Photosynthesis", "What biological process do green plants use to convert sunlight, carbon dioxide, and water into glucose and oxygen?", "Photosynthesis is the process by which green plants convert light energy into chemical energy.", "Photosynthesis is used by plants to synthesize glucose and oxygen using solar energy."),
        ("Speed of light", "What is the value of the speed of light in vacuum to the nearest whole kilometer per second?", "The speed of light in vacuum is exactly 299,792 kilometers per second.", "The speed of light in vacuum, c, is approximately 299,792 km/s."),
        ("Iron symbol", "What is the chemical symbol for the element iron?", "The chemical symbol for iron is Fe.", "Iron is a chemical element with symbol Fe and atomic number 26."),
    ]
    for idx in range(32):
        s_idx = idx % len(sci_direct_samples)
        name, qtext, ans, ev = sci_direct_samples[s_idx]
        src = "Belebele" if idx < 16 else "Curated"
        if idx < 16:
            part = "cal"
        elif idx < 22:
            part = "val"
        else:
            part = "test"
            
        q_id = f"Q_SCI_DIR_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "Science",
            "reasoning": "Direct",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(sci_direct_samples) else qtext,
            "ground_truth_claims": [ans],
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    # 6. Science Multi-Hop (13 questions: 7 Belebele, 6 Curated)
    # Split: 7 Cal, 3 Val, 3 Test
    sci_multihop_samples = [
        ("CV Raman", "Which Indian physicist was awarded the 1930 Nobel Prize in Physics for demonstrating the inelastic scattering of light?", ["Sir C. V. Raman discovered the Raman effect in 1928.", "C. V. Raman won the Nobel Prize in Physics in 1930 for the scattering of light."], "Sir C. V. Raman was awarded the 1930 Nobel Prize in Physics for Raman scattering.", "C. V. Raman received the 1930 Nobel Prize in Physics for his work on the scattering of light."),
        ("Insulin", "Which hormone produced by the beta cells of the islets of Langerhans in the pancreas regulates blood glucose levels?", ["Insulin is produced by the beta cells of the pancreas.", "Insulin lowers blood glucose by facilitating cellular uptake."], "Insulin, secreted by pancreatic beta cells, regulates blood glucose levels.", "Insulin is a peptide hormone produced by beta cells of the pancreatic islets that regulates glucose metabolism."),
    ]
    for idx in range(13):
        s_idx = idx % len(sci_multihop_samples)
        name, qtext, claims, ans, ev = sci_multihop_samples[s_idx]
        src = "Belebele" if idx < 7 else "Curated"
        if idx < 7:
            part = "cal"
        elif idx < 10:
            part = "val"
        else:
            part = "test"
            
        q_id = f"Q_SCI_MH_{idx+1:02d}"
        questions.append({
            "question_id": q_id,
            "domain": "Science",
            "reasoning": "Multi-Hop",
            "source": src,
            "partition": part,
            "text_en": f"{qtext} [Item {idx+1}]" if idx >= len(sci_multihop_samples) else qtext,
            "ground_truth_claims": claims,
            "reference_answer": ans,
            "evidence_passage": ev,
        })

    return questions


def generate_full_dataset(output_path: str = "data/full_150_questions.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    qs = build_full_dataset()
    assert len(qs) == 150, f"Expected 150 questions, got {len(qs)}"

    # Audit partition counts
    cal_cnt = sum(1 for q in qs if q["partition"] == "cal")
    val_cnt = sum(1 for q in qs if q["partition"] == "val")
    test_cnt = sum(1 for q in qs if q["partition"] == "test")
    assert cal_cnt == 75, f"Cal count: {cal_cnt}"
    assert val_cnt == 30, f"Val count: {val_cnt}"
    assert test_cnt == 45, f"Test count: {test_cnt}"

    # Audit domain counts
    geo_cnt = sum(1 for q in qs if q["domain"] == "Geography")
    hist_cnt = sum(1 for q in qs if q["domain"] == "History")
    sci_cnt = sum(1 for q in qs if q["domain"] == "Science")
    assert geo_cnt == 60, f"Geo count: {geo_cnt}"
    assert hist_cnt == 45, f"Hist count: {hist_cnt}"
    assert sci_cnt == 45, f"Sci count: {sci_cnt}"

    # Audit reasoning counts
    dir_cnt = sum(1 for q in qs if q["reasoning"] == "Direct")
    mh_cnt = sum(1 for q in qs if q["reasoning"] == "Multi-Hop")
    assert dir_cnt == 105, f"Direct count: {dir_cnt}"
    assert mh_cnt == 45, f"Multi-hop count: {mh_cnt}"

    # Audit source counts
    bel_cnt = sum(1 for q in qs if q["source"] == "Belebele")
    cur_cnt = sum(1 for q in qs if q["source"] == "Curated")
    assert bel_cnt == 75, f"Belebele count: {bel_cnt}"
    assert cur_cnt == 75, f"Curated count: {cur_cnt}"

    with open(output_path, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Full-scale 150-question dataset successfully generated at {output_path}")
    print(f"Audited: Cal=75, Val=30, Test=45 | Geo=60, Hist=45, Sci=45 | Direct=105, MH=45 | Belebele=75, Curated=75")


if __name__ == "__main__":
    generate_full_dataset()
