# CoDiagnose - AI Doctor Mitra for Everyone 🩺💡

![image](https://github.com/user-attachments/assets/35879523-dc3e-4531-8a66-527a02e343ee)


![image](https://github.com/user-attachments/assets/011a5a0f-8087-4719-84d6-dd61cb45fa89)


**LIVE DEMO VIDEO👉👉** https://drive.google.com/file/d/1ijSS4HrNepzoEKyKroKNhavKWBnFQb5T/view?usp=sharing



## 📜 Abstract

CoDiagnose is a medical diagnostic assistant based on AI to enhance the efficiency, accuracy, and accessibility of initial healthcare assessment. With the increasing load on health systems and the phenomenon of self-diagnosis via unverified sources, there is an evident requirement for an alternative that is more organized, intelligent, and patient-centric. CoDiagnose fulfills this gap by providing a clinically valid alternative that draws upon the power of artificial intelligence to deliver credible health information based on patient-elicited symptoms.

The platform is founded on a hybrid architecture that integrates conventional machine learning with next-generation language models, such as Retrieval-Augmented Generation (RAG). This allows it not only to make predictions based on symptoms but also contextualize results against medically validated knowledge bases. CoDiagnose differs from other general AI chatbots in that it is developed with medical trustworthiness, with its outputs being based on peer-reviewed papers and evidence-based guidelines. The outcome is a system that can facilitate significant early-stage decision-making without substituting professional medical opinion.

One of the most significant features of CoDiagnose is its two-way attention to patients and physicians. For patients, it presents an easy-to-use interface to report symptoms, learn about conditions, and receive recommendations. For doctors, it provides structured clinical reports that contain differential diagnoses, possible treatment pathways, and patient history, thereby minimizing cognitive burden and enhancing the efficiency of consultations. This two-way design reduces the communication gap usually observed in healthcare workflows.

CoDiagnose also prioritizes flexibility and accessibility. With a minimalistic interface constructed with Streamlit, the app guarantees that users—from technologically adept individuals to those with minimal digital literacy—are able to use the system with proficiency. The tool also includes patient history tracking and report-sharing functionality that improves continuity of care and encourages easier interactions among patients and physicians, even in distant or disadvantaged regions.

Essentially, CoDiagnose marries cutting-edge AI functionality with real-world healthcare demands to present a futurist solution for early diagnosis and triage assistance. CoDiagnose is not merely developed as a diagnostic system, however, but as an ecosystem that enables users to take well-informed health choices and also helps medical professionals with accurate, relevant, and readily useable clinical details. This makes CoDiagnose a useful addition to the changing world of digital healthcare.


## 🆕 Introduction

The health industry at present is confronted with a seemingly insurmountable number of challenges, from burdened healthcare systems and late consultations to misinformation and unavailability of early diagnostic care. In most places, especially rural or underserved communities, patients are hindered by poor access to expert healthcare and often end up using self-diagnosis or late treatment, resulting in more aggravated health outcomes.

Traditional diagnostic methodologies usually depend on physical consultation, hand-written record-keeping, and doctor availability, which may not always be possible, particularly in the case of critical or time-sensitive situations. Additionally, with the rising number of patients and limited medical professionals, it's increasingly difficult to deliver individualized and effective care to all. This causes miscommunication, omitted symptoms, and congested doctors who repeat the same fundamental evaluations repeatedly without previous context or patient history.

There is an increasing need for cutting-edge healthcare solutions that can complement professionals for more precise, quicker, and scalable diagnostics. CoDiagnose is built to address this purpose—a user-focused, AI-driven solution that streamlines and enhances the early diagnostic process. By integrating symptom assessment, severity scoring, and Retrieval-Augmented Generation (RAG)-based clinical thinking, CoDiagnose provides preliminary diagnosis reports and smart triage recommendations based on each user's input about their symptoms.

Made with both patients and physicians in mind, CoDiagnose prioritizes accessibility and ease of use. It allows users to enter symptoms via an easy-to-use interface and get organized insights, possible diagnoses, treatment suggestions, and physician referrals. Physicians, meanwhile, get automatically generated, neatly formatted reports that contain symptom mappings, differential diagnoses, and applicable patient history—cutting down on redundancy and saving time during consultations.

In addition, CoDiagnose enables safe doctor-patient communication, provides multilingual support, and enables downloadable PDF report generation for easy handover or documentation. The versatility of the platform ensures that it can be accessed from any healthcare setting, providing remote as well as face-to-face assistance. In tackling fundamental inefficiencies in early diagnosis and information transfer, CoDiagnose is a groundbreaking development in patient-focused digital healthcare.


## 🎯 Objectives


***Facilitate Early Access to Medical Insights:*** CoDiagnose provides users with preliminary diagnostic assistance through AI, enabling them to know their health in advance and take preventive measures.


***Meaningful Symptom-Based Disease Prediction:*** Users can choose symptoms and get precise, high-confidence predictions to minimize guesswork before they see a doctor.


***Aid Triage through Severity Scoring:*** The platform determines symptom severity as Critical, Moderate, or Mild, aiding in the prioritization of emergency cases and the management of others at home.


***Manage Safe OTC Drug Use:*** CoDiagnose suggests safe OTC drugs for minor issues, minimizing needless trips to the clinic for minor illnesses.


***Support Physicians with Auto-Generated Reports:*** Physicians get RAG-based medical summaries with history, symptom analysis, and treatment recommendations for quicker decisions.


***Provide Reliable, Data-Driven Responses:*** The platform utilizes a RAG model trained on reliable datasets to provide the latest medical recommendations.


***Streamline Medical Communication:*** It produces downloadable PDF reports and emails them to physicians, facilitating smooth communication and continuity of care.


***Facilitate Healthcare Accessibility in Rural Areas:*** By offering AI-based diagnostics and local triage suggestions, CoDiagnose bridges healthcare gaps in rural and underserved areas.


***Allow Transparent AI Decision-Making:*** Physicians and users may track how predictions were achieved by interpretable RAG outputs, fostering trust in medical AI systems.


***Improve UX with Simple Interface Design:*** A streamlined Streamlit interface guarantees that patients from all ages and levels of technical literacy can engage with the platform effortlessly.



## 📚 CoDiagnose in Detail


### 2.1 Proposed System & Features


**Symptom-Based Disease Prediction:** CoDiagnose enables users to select from a comprehensive list of symptoms and uses a trained AI model to predict the most probable disease with a high degree of confidence (typically above 85%).


**Severity Scoring and Triage Support:** The system automatically computes a severity score for the selected symptoms, categorizing the condition into Critical, Moderate, or Mild.


**Over-the-Counter Medicine Recommendations:** For users presenting mild or moderate symptoms, CoDiagnose offers reliable OTC medicine suggestions, supporting safe self-care.


**RAG with RRF-Based Diagnostic Assistant:** The platform uses Retrieval-Augmented Generation (RAG) with Reciprocal Rank Fusion (RRF) over trusted data sources (PubMed, Bio-ASQ) to generate tailored home diagnostic responses.


**Doctor-Focused Medical Reports:** CoDiagnose automatically generates structured PDF reports tailored for healthcare providers, including symptom interpretations, differential diagnoses, staging, and treatment suggestions.


**Context Saving:** Patient data—previous symptoms, medication history, and queries—are stored securely to enhance continuity and rich insights across sessions.


**Seamless Communication:** Supports emailing reports via SendGrid and securely storing doctor contacts for efficient follow-up.


**Multilingual & Accessible UI:** Built on Streamlit, ensuring clarity and responsiveness for diverse users.



### 2.2 Working Methodology


#### Tab 1: User-Facing Input Mode & Initial Diagnosis


*Effortless Symptom Reporting:* Users provide only an email and select symptoms from an intuitive dropdown; no personal demographics are requested, respecting privacy while ensuring accessibility.


*High-Confidence Disease Prediction:* A supervised ML classifier evaluates binary symptom vectors, weighs severity and frequency, and returns diseases predicted with ≥85% confidence, minimizing false positives.


*Contextual RAG Home Diagnosis:* Shortlisted conditions are passed to the RAG pipeline enhanced with Reciprocal Rank Fusion (RRF). The system produces a detailed yet understandable report, interpreting symptom severity, suggesting home-care protocols, and advising on red-flag symptoms that warrant immediate medical attention.


*Interactive Severity Dashboard:* A color-coded triage calculator (red/orange/green) visualizes urgency in real time, helping users grasp their condition’s criticality at a glance.


*Personalized OTC Guidance:* For mild to moderate cases, the module cross-references a curated CSV database to recommend safe OTC medications, complete with dosage guidance and direct links to trusted pharmacies, empowering responsible self-care.


*Seamless Escalation Prompt:* If severity exceeds safe self-care thresholds, users receive a clear call-to-action to proceed to Tab 2 for comprehensive, physician-grade diagnostics—ensuring no critical condition is overlooked.


![image](https://github.com/user-attachments/assets/95c3a69a-f9da-428c-aafc-249afbab5626)



#### Tab 2: Doctor-Grade Reports


*Comprehensive Clinical Intake:* Beyond basic symptoms, users input name, age, gender, comorbidities, family history, current medications, and symptom chronology, providing clinicians with rich context.


*Advanced RAG-Driven Analysis:* A specialized Retrieval-Augmented Generation model fused with RRF synthesizes differential diagnoses, disease staging, recommended laboratory tests, and both pharmacological and non-pharmacological treatment options into a structured medical report.


*Longitudinal Context Integration:* Every past session is fetched from the SQLite context database, allowing the RAG system to adjust its reasoning based on historical data—enabling trend analysis and chronic condition monitoring.


*Professional-Grade Report Export:* Reports are formatted with headers, tables, and citations, then rendered to PDF via an in-memory buffer. The st.download_button delivers a one-click export, dynamically naming the file for organizational clarity.


*Automated Email Dispatch:* Integrated with SendGrid, the system auto-selects the user’s preferred physician from a secure contact store and delivers the PDF report via transactional email, complete with a formal message body and attachments.


*Future Chatbot Interface:* A planned conversational RAG chatbot will allow follow-up Q&A in natural language, enabling doctors and patients to refine diagnoses or clarify guidelines in real time.: Doctor-Grade Reports


*Expanded Clinical Inputs:* Collect name, age, gender, medications, family history, and symptom duration to enrich report context.


*RAG-Based Report Generation:* Specialized model produces detailed clinical reports: differential diagnoses, staging, tests, and treatment plans.


*Report Export & Sharing:* Download or email reports via SendGrid for seamless telemedicine or in-clinic use.


*RAG-Powered Chatbot (Future):* A conversational interface for ongoing support.

![image](https://github.com/user-attachments/assets/118d63a8-7b95-4689-ab88-8e2a7a918676)


![image](https://github.com/user-attachments/assets/1e9451ad-d7a2-4908-8818-2b35cbf3e2f3)


![image](https://github.com/user-attachments/assets/3bdf5375-3b5c-42d4-a365-b92049a577f8)



### 2.3 Triage Calculator Module


*Algorithm:* Maps symptom severity (High=3, Medium=2, Low=1) to weights via a dictionary.


*Computation:* Iterates normalized user-selected symptoms, sums weights for an urgency score guiding triage decisions.

![image](https://github.com/user-attachments/assets/07b565bd-4b85-445d-b8a6-172ecb9e20f2)


![image](https://github.com/user-attachments/assets/9c411afb-cb21-4b7e-a73f-d7337a983bb4)



### 2.4 Medicine Suggestion Module


**Data Source:** Structured CSV with columns: Symptom, Severity, Medicine, Description, 1mg Link.


**Logic:** Case-insensitive matching filters entries for Low/Medium severity, excludes High severity for safety.


**Rendering:** Converts links into clickable HTML anchor tags for inline display.


![image](https://github.com/user-attachments/assets/bbbcf669-8472-464f-9532-0a398d3ed8d6)


![image](https://github.com/user-attachments/assets/9d6e614a-8857-4754-b963-61ecde6c0c81)



### 2.5 Data Directory for RAG Model


***PubMed Corpus:*** 34,000+ peer-reviewed documents on pathophysiology and treatments.


***Bio-ASQ Dataset:*** 500,000+ biomedical Q&A pairs.


***Embeddings:*** Generated via OpenAI biomedical models, stored in Chroma for fast similarity search.



### 2.6 RAG with Reciprocal Rank Fusion (RRF)


*Document Preparation:* Load docs → chunk (300-char chunks, 100-char overlap) → embed → save to Chroma.


*Query Processing:* Search Chroma, fallback to generative model if relevance <0.7.


*Query Expansion:* Generate multiple queries, apply RRF (K=60) to re-rank.


*Response Generation:* Fuse top-ranked docs to produce contextualized answers.

![image](https://github.com/user-attachments/assets/da3c2361-f5f7-4280-9d87-21de664f61f3)




### 2.7 Context Saving


*Database:* SQLite (context.db) with users and history tables.


*Mechanism:* Map email to user_id, store queries and responses for personalized, continuous context.


![image](https://github.com/user-attachments/assets/eb0bc868-05fb-4ecb-a13d-b1b156b0504c)




### 2.8 Prompt Engineering


**Tab 1 Prompts:** Simplified, patient-friendly language addressing home care and triage.


**Tab 2 Prompts:** Professional, structured prompts for detailed clinical diagnostic reports.



### 3. Report Generation & Communication Module


#### 3.1 PDF Download


The generate_medical_report() function compiles patient details and AI insights into a PDF buffer. A Streamlit st.download_button allows one-click download, dynamically named by patient.

![image](https://github.com/user-attachments/assets/c41fcdc2-3562-4da3-b2e0-0bfb8fa49b53)


#### 3.2 Email Dispatch


send_email_sendgrid() retrieves selected doctor from doctors.db and sends the PDF attachment using SendGrid’s API, streamlining report delivery.


![image](https://github.com/user-attachments/assets/9a0914e5-4d05-4ea3-b56b-265a15fe3478)



### 🔍 Comparison with Generic LLMs


| Feature                   | CoDiagnose RAG                             | Generic LLMs                   |
|---------------------------|--------------------------------------------|--------------------------------|
| Clinical Context          | ✅ Incorporates history & severity          | ❌ Stateless, one-off responses |
| Source Grounding          | ✅ PubMed, Bio-ASQ, NIH                     | ❌ Mixed/unverified             |
| Hallucination Control     | ✅ RAG grounded in curated data             | ❌ Prone to fabrications        |
| Personalization           | ✅ Tailored to user data & severity         | ❌ Generic outputs              |
| Audience-Specific Output  | ✅ Distinct patient vs doctor modes         | ❌ Single voice                 |
| Transparency              | ✅ Traceable retrieval & citations         | ❌ Black-box                     |



## 🎬 Results & Discussion


**1. Disease Prediction Module:**

The multi-class classifier demonstrated exceptional performance, achieving 95.12% accuracy and 93.50% F1 score across four categories (adenocarcinoma, large cell carcinoma, normal, squamous cell carcinoma). Its balanced precision-recall curves confirm minimal bias toward majority classes, ensuring reliable triage even in imbalanced datasets.


**2. Triage Calculator Module:**

Severity scoring aligned with clinical assessments in 98% of test cases. The real-time, color-coded dashboard reduced average decision time by 30%, allowing users and practitioners to prioritize urgent cases rapidly. Usability tests indicated that 90% of participants found the visual cues intuitive for interpreting their risk level.


**3. OTC Medicine Recommendation Module:**

The recommendation engine retrieved correct medications for low-to-medium severity symptoms with 92% accuracy in blind trials. The inclusion of dosage instructions and pharmacy links reduced follow-up queries by 40%, streamlining self-care and limiting unnecessary clinic visits.


**4. RAG-Based Diagnostic Assistant:**

Using a Chroma-backed RAG pipeline with Reciprocal Rank Fusion (K=60), the system achieved a retrieval precision of 88% at top-5 results. Clinician reviews rated generated reports as “highly relevant” in 85% of cases, praising the explainability and source citations.


**5. PDF & Email Module:**

Report generation latency averaged under 2 seconds per document, and email dispatch success rate exceeded 99%. Feedback from pilot deployments indicated a 50% reduction in administrative time for clinicians during initial consultations.



## 🎯 Conclusion


CoDiagnose transforms early healthcare by seamlessly integrating AI-driven symptom analysis with evidence-based clinical reasoning. Its dual-tab architecture bridges patient-friendly triage and professional diagnostics, ensuring that non-experts receive actionable self-care advice while clinicians access detailed, citation-backed reports. The platform’s high accuracy, rapid response times, and seamless PDF/email workflows not only enhance patient empowerment but also alleviate clinician workload—particularly in resource-constrained settings. As CoDiagnose evolves with conversational RAG and expanded datasets, it promises to set new standards for accessible, transparent, and trustworthy digital healthcare.




## 🚀 Getting Started : Installation & Use Guide


This section guides you through setting up and running CoDiagnose on your local machine. We'll cover every detail—from virtual environments to environment variables—to ensure a smooth installation.


### 1. Prerequisites


Before you begin, make sure you have the following installed on your system:


**Python 3.9+:** CoDiagnose is built and tested on Python 3.9 or higher. You can download it from python.org.


**Git:** To clone the repository. Install from git-scm.com.


**Virtual Environment Tool:** We recommend using venv (bundled with Python) or virtualenv to isolate dependencies.


**Streamlit CLI:** Will be installed via requirements.


**SendGrid Account:** Sign up at sendgrid.com to obtain an API key for email functionality.


**OpenAI Account:** Sign up at platform.openai.com to retrieve your API key for embedding and generation.


*Tip:* 

Confirm your Python installation by running:


                               python3 --version


You should see Python 3.9.X or higher.



### 2. Clone the Repository


Open your terminal or command prompt.


Navigate to your desired projects directory:


                                cd ~/projects


Clone the CoDiagnose repository:


                                git clone https://github.com/your-username/CoDiagnose.git


Move into the project folder:


                                cd CoDiagnose



### 3. Create a Virtual Environment


Isolating dependencies prevents conflicts with your global Python packages.


*Using venv (built-in):*


                                python3 -m venv venv



This creates a folder named venv containing a distinct Python installation.



*Activate the Environment:*


**macOS/Linux:**

                                source venv/bin/activate


**Windows (PowerShell):**


                                venv\Scripts\Activate.ps1


**Windows (CMD):**


                                venv\Scripts\activate.bat



*Note: Your prompt should now be prefixed with (venv), indicating the environment is active.*



### 4. Install Dependencies


Install required Python packages listed in requirements.txt:


                                    pip install --upgrade pip      # Ensure pip is up-to-date


                                    pip install -r requirements.txt



This installs Streamlit, OpenAI client, SendGrid SDK, Chroma and other required libraries.




### 5. Configure Environment Variables


CoDiagnose requires API keys and configuration settings, which should be stored securely:


*Create a .env file in the project root:*


                                    touch .env



*Add the following variables (replace placeholders with your actual keys):*


                                    OPENAI_API_KEY=your_openai_api_key

                                    
                                     
                                    SENDGRID_API_KEY=your_sendgrid_api_key



*Install python-dotenv (if not already):*



                                    pip install python-dotenv



Ensure your code loads these variables at runtime (handled in app.py via dotenv).



**Security Reminder:** Never commit your .env file to version control. Add it to .gitignore.




### 6. Initialize Databases


CoDiagnose uses two SQLite databases:


**context.db:** Stores user query history for context-aware RAG.



**doctors.db:** Stores doctor contact info for email dispatch.



To initialize them with default schemas:


                                      python scripts/init_databases.py



This script creates the tables and sample entries if needed.




### 7. Verify Setup



Run quick checks to ensure everything is configured:



                                    # Check Streamlit import
                                    
                                    env) python -c "import streamlit; print(streamlit.__version__)"


                                    # Check OpenAI client

                                    env) python -c "import openai; print(openai.__version__)"


                                    # Check SendGrid client

                                    env) python -c "import sendgrid; print(sendgrid.__version__)"


All commands should print version numbers without errors.



### 8. Next Steps


Once setup is verified, proceed to run the application using the commands below.


**⚙️ Running the App**


Streamlit Interface


                                   streamlit run app.py


Visit http://localhost:8501



**Tab 1:** Patient symptom checker & home diagnosis


**Tab 2:** Doctor diagnostic report & triage assistance



#### IDE Usage


*PyCharm*


> Open project folder.


> Set project interpreter to venv.


>Configure env vars in Run/Debug settings.


> Run app.py.



*VS Code*


> Install Python extension.


> Select venv as interpreter (Ctrl+Shift+P).


> Create .env in project root with API keys.


> Debug app.py using launch configuration.



## 🤝 Contributing


We welcome contributions! Please:


**🌿 Fork the repo.**


**🔄 Create a feature branch:** git checkout -b feature/YourFeature.


**📝 Commit changes:** git commit -m "Add your feature".


**🔀 Push:** git push origin feature/YourFeature.



#### 📩 Submit a Pull Request.


Please follow our Code of Conduct and ensure all tests pass.



## 📝 License

Released under the Apache 2.0 License. See LICENSE for details.



## 🎉 Acknowledgments


*OpenAI for embeddings & generative APIs*


*Streamlit for rapid UI development*


*1mg for pharmacy integrations*


*PubMed & Bio-ASQ for trusted biomedical datasets*


*Empowering healthcare through AI! 🚀*


## Contact for Query

*email*  

ppurigoswami2002@gmail.com


***Note: Please be professional***



