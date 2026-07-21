# Student lifestyle & depression risk model

#Problem

Universities and student-wellness teams collect lifestyle data (sleep, study habits, stress, screen time) but rarely turn it into something a student or advisor can act on in the moment. This project asks: given a student's everyday lifestyle numbers, how much do they actually explain depression risk, and can that be delivered as an instant, private, client-side tool?

#Dataset
->100,000 student records, 11 columns: demographics (age, gender, department), academic performance (CGPA), lifestyle (sleep duration, study hours, social media hours, physical activity), stress level (1–10), and a depression flag.
->No missing values. Base rate: 10.1% of students flagged as depressed.

#Exploratory findings that shaped the model
->Stress is not linear. Depression rate rises gently from stress level 2 (8.9%) to level 7 (12.2%), then jumps to ~42% at levels 8–9. Only ~2% of students are in that band, but the relationship is a threshold, not a slope — so a plain linear stress term underfits it badly. This is why the model includes an explicit High_Stress (stress ≥ 8) indicator alongside the continuous stress value.
->CGPA has the strongest single correlation with depression (r ≈ -0.18) of any variable in the dataset.
->Sleep and stress are linked (r ≈ -0.28); students sleeping under 5 hours show roughly double the depression rate of everyone else.
->Department and gender barely matter — depression rates sit within a 0.3-point band across all five departments and both genders.

#Approach
1.Feature engineering: one-hot encode department and gender, add the High_Stress threshold flag described above.
2.Model: logistic regression (scikit-learn), all features standardized, class_weight="balanced" — a deliberate choice, since depression is only ~10% of rows and a health-adjacent risk tool should prioritize catching real cases (recall) over raw accuracy.
3.Split: 80/20 train/test, stratified on the target, random_state=42.
4.Export: the fitted coefficients, intercept, and standardization parameters are copied directly into index.html as a JS object, so the exact same linear model runs client-side with zero inference cost and zero data leaving the browser.

See train_model.py for the full, runnable training pipeline — it prints every number used in the app and writes them to model_export.json.

#Results (held-out test set, 20,000 students)
Metric	Value
Accuracy	63.7%
Precision	17.1%
Recall	67.8%
F1	0.273
ROC-AUC	0.693

#Confusion matrix:

	Predicted: no depression	Predicted: depression
Actual: no depression	11,378	6,610
Actual: depression	647	1,365

The model trades a lot of precision for recall on purpose: it catches roughly two out of three real depression cases, at the cost of a high false-positive rate. For a screening-style tool that's the safer failure mode — missing a real case is worse than an extra false alarm.

#What drives the prediction

Ranked by standardized coefficient magnitude:

1.CGPA (strongest — lower CGPA → higher predicted risk)
2.Crossing the stress-8+ threshold
3.Continuous stress level
4.Sleep duration (more sleep → lower risk)
5.Age, department, gender, study hours, social media, and physical activity all contribute far less.

#Honest limitations
->ROC-AUC of 0.69 means the model beats chance but is nowhere near a reliable individual-level predictor — real depression risk depends on far more than eight lifestyle numbers.
->The dataset appears synthetic/survey-style; relationships here may not generalize to a real student population.
->This is a portfolio demonstration, not a diagnostic or clinical tool. The app states this explicitly and points anyone concerned about their own mental health toward a real person — a doctor, counselor, or trusted contact.

#Files
index.html — the self-contained web app (form, live prediction, risk gauge, distribution comparison chart, model documentation)
train_model.py — reproducible training script; run it against the raw CSV to regenerate every number in this README and in the app
model_export.json — the trained model's coefficients and metrics in a machine-readable form
