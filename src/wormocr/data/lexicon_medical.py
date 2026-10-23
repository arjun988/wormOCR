"""Prescription-style vocabulary for synthetic handwriting.

Generic medicine names (common international nonproprietary names), dosage
forms, and standard prescription abbreviations. No patient data.
"""

GENERICS = """
paracetamol acetaminophen ibuprofen aspirin diclofenac naproxen tramadol morphine codeine
amoxicillin ampicillin azithromycin clarithromycin doxycycline ciprofloxacin levofloxacin
metronidazole cefixime ceftriaxone cefuroxime cephalexin clindamycin nitrofurantoin
cotrimoxazole fluconazole clotrimazole ketoconazole acyclovir oseltamivir albendazole
ivermectin omeprazole esomeprazole pantoprazole rabeprazole ranitidine famotidine domperidone
ondansetron metoclopramide loperamide lactulose bisacodyl simethicone metformin glimepiride
gliclazide sitagliptin insulin amlodipine atenolol metoprolol bisoprolol propranolol losartan
telmisartan olmesartan valsartan enalapril ramipril lisinopril hydrochlorothiazide furosemide
spironolactone atorvastatin rosuvastatin simvastatin clopidogrel warfarin heparin
levothyroxine prednisolone prednisone dexamethasone hydrocortisone methylprednisolone
salbutamol montelukast budesonide fluticasone cetirizine levocetirizine loratadine
fexofenadine chlorpheniramine diphenhydramine amitriptyline sertraline fluoxetine
escitalopram alprazolam clonazepam diazepam lorazepam zolpidem gabapentin pregabalin
carbamazepine levetiracetam phenytoin valproate haloperidol olanzapine risperidone
quetiapine donepezil folic acid ferrous sulfate calcium carbonate vitamin cholecalciferol
cyanocobalamin thiamine pyridoxine zinc magnesium potassium chloride sodium bicarbonate
oral rehydration salts dextrose saline mupirocin fusidic betamethasone tamsulosin
finasteride sildenafil misoprostol oxytocin progesterone estradiol tranexamic baclofen
methocarbamol colchicine allopurinol febuxostat hydroxychloroquine methotrexate
""".split()

FORMS = [
    "Tab", "Tab.", "Cap", "Cap.", "Syr", "Syr.", "Inj", "Inj.", "Susp", "Oint", "Cream",
    "Drops", "Gel", "Inhaler", "Sachet", "Lotion", "Neb", "IV", "IM", "SC",
]

FREQUENCIES = [
    "OD", "BD", "BID", "TDS", "TID", "QID", "QDS", "HS", "SOS", "PRN", "STAT",
    "1-0-1", "1-1-1", "0-0-1", "1-0-0", "0-1-0", "1/2-0-1/2", "a.c.", "p.c.", "h.s.",
    "once daily", "twice daily", "at night", "after food", "before food", "as needed",
]

DURATIONS = ["x 3 days", "x 5 days", "x 7 days", "x 10 days", "x 2 wks", "x 1 month", "for 5 days", "1 week"]

UNITS = ["mg", "mg", "mg", "mcg", "g", "ml", "mL", "IU", "%"]

NOTES = [
    "Rx", "Dx", "Hx", "c/o", "k/c/o", "BP", "Pulse", "Temp", "SpO2", "Wt", "Ht", "RBS", "FBS",
    "CBC", "LFT", "RFT", "ECG", "X-ray", "USG", "Review after", "Follow up", "Adv", "Plenty of fluids",
    "Avoid oily food", "Rest", "Fever", "Cough", "Cold", "Headache", "Pain abdomen", "Vomiting",
]
