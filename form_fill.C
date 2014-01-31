void form_fit() { 
 
 // Define formula with free parameters for fitting 
 tf = new TF1("tf", "[0]*exp([1]*x)", 0, 10); 
 tf->SetLineColor(kRed); 
 
 tf2 = new TF1("tf2", "[0]*x^2 + [1]*x + [2]", 0, 5); 
 tf2->SetParameters(-1, 1, -5); 
 tf2->SetLineColor(kBlue); 
 
 // Define "data" plot to fit 
 h = new TH1F("h", "Histogram", 10, 0, 10); 
 for(int i=0; i<100; i++) h->Fill( gRandom->Exp(3) ); 
 
 // Graph 
 tg=new TGraph(); // can use arrays, text files, or SetPoint to fill 
 tg->SetPoint(0, 1.0, 2.0); // start with point #0 
 tg->SetPoint(1, 2.0, 7.0); 
 tg->SetPoint(2, 3.0, 13.0); 
 tg->SetPoint(3, 4.0, 12.0); 
 tg->SetPoint(4, 5.0, 6.5); 
 
 
 // Now handle the drawing and formatting 
 c1 = new TCanvas("c1", "My Plots", 800, 400); 
 c1->Divide(2,1); 
 
 // *** Panel 1 *** 
 c1->cd(1); 
 h->Draw(); 
 h->Fit(tf); 
 
 // *** Panel 2 *** 
 c1->cd(2); 
 tg->SetTitle("Graph"); 
 tg->SetMarkerStyle(20); 
 tg->SetMarkerColor(kRed); 
 tg->Draw("AP"); 
 tg->Fit(tf2); 
 
}