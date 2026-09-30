// explore_root_file.cpp
// Compilazione/Esecuzione in ROOT: root -l -b -q 'explore_root_file.cpp("path/to/file.root")'

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/InterfaceUtils.hxx>
#include <TCanvas.h>
#include <TH1D.h>
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <vector>
#include <string>

void explore_root_file(const std::string& filename, const std::string& tree_name = "AnalysisMiniTree") {
    // Abilita il multithreading automatico
    ROOT::EnableImplicitMT();

    // 1. Ispezione del Tree
    TFile file(filename.c_str(), "READ");
    TTree* tree = (TTree*)file.Get(tree_name.c_str());
    if (!tree) {
        std::cerr << "Errore: TTree " << tree_name << " non trovato." << std::endl;
        return;
    }

    std::cout << "========================================================================\n";
    std::cout << "TTree: " << tree_name << "\n";
    std::cout << "Numero di eventi: " << tree->GetEntries() << "\n";
    std::cout << "Numero di branch: " << tree->GetListOfBranches()->GetEntries() << "\n";
    std::cout << "========================================================================\n";

    // 2. Inizializzazione RDataFrame
    ROOT::RDataFrame df(tree_name, filename);

    // Lista dei branch chiave da analizzare
    std::vector<std::string> key_branches = {
        "recojet_antikt4PFlow_pt___NOSYS",
        "recojet_antikt4PFlow_eta",
        "tau_pt___NOSYS",
        "tau_RNNJetScoreSigTrans",
        "bbtt_HH_m___NOSYS"
    };

    // 3. Elaborazione e Grafica
    for (const auto& branch : key_branches) {
        // Verifica esistenza del branch
        if (!df.HasColumn(branch)) {
            std::cout << "Branch " << branch << " non presente nel TTree. Skip.\n";
            continue;
        }

        // Estrazione statistiche di base tramite RDataFrame
        auto mean = df.Mean(branch);
        auto stddev = df.StdDev(branch);
        auto min_val = df.Min(branch);
        auto max_val = df.Max(branch);

        std::cout << "Branch: " << branch << "\n"
                  << "  Mean: " << *mean << " | StdDev: " << *stddev
                  << " | Min: " << *min_val << " | Max: " << *max_val << "\n";

        // Creazione e salvataggio dell'istogramma
        auto hist = df.Histo1D({branch.c_str(), branch.c_str(), 60, *min_val, *max_val}, branch);
        
        TCanvas canvas("c", "c", 800, 600);
        hist->DrawClone();
        std::string out_name = "dist_" + branch + ".png";
        canvas.SaveAs(out_name.c_str());
        std::cout << "  -> Istogramma salvato: " << out_name << "\n\n";
    }
}