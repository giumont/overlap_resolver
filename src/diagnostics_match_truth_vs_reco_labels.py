import uproot
import awkward as ak

# Caricamento del TTree
file_path = "HH_bbtt/output_GGF_mc23a_bypass_noOR_000001.root" # Inserire il percorso reale del file
tree = uproot.open(f"{file_path}:AnalysisMiniTree") #

# --- Lettura branch Tau ---
# Array Reco e Truth pre-allineato
tau_reco_pt = tree["tau_pt___NOSYS"].array()
tau_truth_aligned_pt = tree["tau_truth_pt_vis"].array()

# Array Truth indipendente[cite: 1]
truthtau_pt = tree["truthtau_pt_vis"].array()

# --- Lettura branch Jet ---
# Array Reco e Truth pre-allineato[cite: 1]
jet_reco_pt = tree["recojet_antikt4PFlow_pt___NOSYS"].array()
jet_truth_aligned_label = tree["recojet_antikt4PFlow_HadronConeExclTruthLabelID"].array()

# Array Truth indipendente[cite: 1]
truthjet_pt = tree["truthjet_antikt4_pt"].array()

# --- Esecuzione della diagnostica ---
print("=== Diagnostica Tau ===")
# Verifica che la lunghezza dell'array per ogni evento sia identica tra reco e truth allineato
tau_aligned_match = ak.all(ak.num(tau_reco_pt) == ak.num(tau_truth_aligned_pt))
print(f"Le dimensioni per evento di tau_pt___NOSYS e tau_truth_pt_vis corrispondono: {tau_aligned_match}")

# Verifica che le collezioni indipendenti abbiano lunghezze diverse
tau_indep_diff = ak.any(ak.num(tau_reco_pt) != ak.num(truthtau_pt))
print(f"Le dimensioni per evento tra tau_pt___NOSYS e truthtau_pt_vis differiscono: {tau_indep_diff}")

print("\n=== Diagnostica Jet ===")
# Verifica allineamento tra jet reco e label di truth pre-calcolate
jet_aligned_match = ak.all(ak.num(jet_reco_pt) == ak.num(jet_truth_aligned_label))
print(f"Le dimensioni per evento di recojet_antikt4PFlow_pt___NOSYS e recojet_antikt4PFlow_HadronConeExclTruthLabelID corrispondono: {jet_aligned_match}")

# Verifica divergenza dimensionale con la collezione truth independente
jet_indep_diff = ak.any(ak.num(jet_reco_pt) != ak.num(truthjet_pt))
print(f"Le dimensioni per evento tra recojet_antikt4PFlow_pt___NOSYS e truthjet_antikt4_pt differiscono: {jet_indep_diff}")