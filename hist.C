//Histogram
//Pattern Rec Spring 2014
//Travis Cramer

void hist() {
	h = new TH1D("h", "Hist", 100, 0, 10);
	h->Fill(2);
	h->Fill(2);
	h->Fill(3);
	h->Fill(4);

	for(i = 0; i < 1000; i++) 
		h->Fill(gRandom->Gaus(5, 1.5));

	h->SetFillColor(kBlue-8);

	for (i=1; i<=h->GetNbinsX(); i++) 
		cout << "Bin " << 1 << "\t" << h->GetBinContent(i) << endl;

	h->SetBinContent(50, 999);
	
	f = new TF1("f", "x", 0, 10);
	h->Eval(f);

	h->Draw();
}