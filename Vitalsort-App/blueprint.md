Master Blueprint: Vitalsort

System Overview The system, provisionally titled Vitalsort, is a multimodal diagnostic triage framework. It integrates concurrent visual and textual data streams to generate algorithmic clinical support, addressing the context limitations of single-modality medical applications.
1. Core System Architecture (Functional Pillars)
The Minimum Viable Product (MVP) is structured around five functional pillars:
Pillar 1: Multimodal Data Fusion and Intake Framework
Visual Ingestion Interface: This secure module is designed for the ingestion of high-resolution macro-photographs of localized clinical signs. It incorporates real-time edge-detection and luminance validation to ensure image quality.
Dynamic Natural Language Intake: A text-parsing engine utilizes Natural Language Processing (NLP) to conduct a semi-structured intake query. It dynamically adjusts follow-up prompts based on initial keywords to capture systemic attributes like chronicity, severity, and associated symptoms.
Anamnesis Synthesis: A voluntary structured input field captures the subject's relevant medical history, comorbidities, and current pharmaceutical profiles to weight the diagnostic algorithms.
Pillar 2: Automated Differential Triage and Inference Engine
Differential Diagnosis Classifier: The system evaluates visual vectors and text tokens against a clinical database to output a probabilistically ranked list of the top three differential conditions.
Explainable AI (XAI) Architecture: The engine generates post-hoc explainability metrics to counter deep learning "black box" limitations. The UI maps visual regions of interest (via heatmaps or bounding boxes) and highlights key verbal tokens contributing to the inference.
Critical Heuristic Triage (Red Flag Protocols): An un-bypassable heuristic override detects high-acuity indicators like systemic sepsis markers or acute anaphylactic language. When triggered, it initiates an immediate systemic halt and displays standardized emergency care protocols.
Pillar 3: Clinical Referral and Urgency Stratification Protocol
Specialist Categorization Matrix: An internal lookup framework maps primary differential classifications to corresponding medical sub-specialties.
Clinical Acuity Stratification: A tri-tier urgency model (Routine, Urgent, Emergency) guides user behavior based on the statistical virulence and progression velocity of the inferred conditions.
Pillar 4: Standardized Pre-Consultation Reporting Interface
Clinical Dossier Generation: The system compiles user inputs, imagery, and triage outputs into a standardized, interoperable PDF or secure cryptographic payload.
Triage Document Structure: Formatted according to clinical summary conventions, this allows clinicians to quickly review data and optimize patient intake.
Pillar 5: Data Governance, Ethics, and Privacy Architecture
Zero-Retention Image Ingestion: Imagery is processed entirely in volatile memory via secure web-sockets and permanently purged post-inference to ensure privacy compliance (e.g., HIPAA, GDPR).
Anonymized Session Protocols: The MVP supports an unauthenticated operating tier for baseline triage without generating or persisting Personally Identifiable Information (PII).
Non-Diagnostic Liability Safeguards: A strict ante-hoc legal acknowledgment framework explicitly defines outputs as probabilistic triage data rather than definitive, legally binding clinical diagnoses.

2. Data Ingestion and Preprocessing
Computer Vision Preprocessing: Protocols include resizing, aspect ratio normalization, and color space conversion (e.g., RGB to HSV/YCbCr). Algorithmic assessment of image sharpness (Laplacian variance) and ambient luminance boundaries reject poor-quality inputs.
NLP Intake Module: Text normalization includes tokenization, lemmatization, and stop-word removal. Specialized biomedical language models (e.g., BioBERT) are implemented for Clinical Named Entity Recognition (NER) to extract medical concepts.

3. Multimodal Data Fusion Mechanism
Vitalsort employs an Intermediate Fusion Strategy utilizing a transformer-based cross-attention mechanism. This projects visual features and textual semantics into a shared latent embedding space.
Cross-Modal Alignment Mathematics:
The visual feature map extracted from the Vision Transformer (ViT) backbone is defined as .
The textual feature sequence from the BioBERT NLP encoder is defined as .
The system computes the Query $(Q)$ from the textual features, while the Key  and Value  matrices are derived from the visual features.
The linear projections are calculated as follows:   
The scaled dot-product attention dictates which image regions are clinically relevant to the text: 
The fused representation passes through Layer Normalization and a Multi-Layer Perceptron (MLP) block:  
The final high-dimensional vector, $F_{final}$, is passed to a classification layer where a softmax activation function computes the probabilistic distribution of the top-k differential diagnoses.
________________________________________________________________________________________

AI Operational Rules & Developer Guidelines
1. Strict Preservation of Progress

No Total Rewrites: You are strictly forbidden from rewriting the entire file or project.

Zero Regression: Always retain the user's existing output progress. Do not delete or overwrite functioning code unless specifically commanded to do so.

2. Architectural & UI Integrity

Do Not Redesign: Never alter the existing User Interface (UI) layout, color schemes, or front-end structure unless explicitly requested.

Maintain System Logic: Preserve all established architectural patterns, variable naming conventions, and state management structures currently in place.

Surgical Precision: Only change, add, or remove exactly what the user explicitly commands.

3. Code Quality & Execution

Concise Implementation: Generate code that is straightforward and easy to use. Avoid over-engineering, convoluted logic, or redundant boilerplate.

Native Preference: If a simpler, native built-in function achieves the goal without requiring heavy external libraries or massive frameworks, use it.

Isolated Outputs: When adding a feature or fixing a bug, output only the specific function, component, or code block that was modified, providing just enough surrounding context for the developer to paste it in securely.

4. Debugging Protocol

Targeted Fixes: If presented with an error message, provide a concise fix that addresses only that specific error without altering the surrounding architecture. Do not attempt to "optimize" adjacent code that is already working.