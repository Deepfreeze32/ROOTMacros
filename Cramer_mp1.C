void Cramer_mp1() {
	float x1[] = {9, 10, 10.7, 11, 11.5, 12, 13, 14, 15}; 
	float y1[] = {0, 0.08, 0.24, .21, 0.15, .18, .30, .17, .05}; 
 	float y2[] = {0.03, .14, .27, .33, .36, .3, .18, .04, 0}; 
	
	//PROBLEM 1
 	c1 = new TCanvas("c1", "2.1", 800, 400);
 	//c1->Divide(2,1);

	tg1 = new TGraph(9,x1,y1);
	tg1->SetMarkerColor(kBlack);
	tg1->SetMarkerStyle(20);
	tg2 = new TGraph(9,x1,y2);
	tg2->SetMarkerColor(kRed);
	tg2->SetMarkerStyle(20);

	tf1 = new TF1("tf1","gaus(0) + gaus(3)", 9, 15);
	tf1->SetParNames("Amp", "Mean", "Width", "Amp", "Mean", "Width");
	tf1->SetParameters(.2, 10.5, 0.5, 0.38, 13.01, 0.5);

	tf2 = new TF1("tf2", "gaus(0) + gaus(3)", 9, 15);
	tf2->SetParNames("Amp", "Mean", "Width", "Amp", "Mean", "Width");
	tf2->SetParameters(0.21,10.8,0.75,0.3,13,3);

	//c1->cd(1);
	tf1->SetLineColor(kBlack);
	
	tf2->SetLineColor(kRed);
	tf2->Draw();
	tf1->Draw("same");
	tg1->Draw("P");
	tg2->Draw("P");
	tg1->Fit(tf1);
	tg2->Fit(tf2);
	
	TLegend leg ( . 1 , . 7 , . 3 , . 9 , "Gaussian Paramters") ;
	leg.SetFillColor(0);
	leg.AddEntry("");

	//PROBLEM 2
	c2 = new TCanvas("c2", "2.2", 800,400);
	//c2->cd();
	th1 = new TH1F("th1", "Histogram", (Int_t)60, (Double_t)9.0, (Double_t)15.0);
	th2 = new TH1F("th2", "Histogram", (Int_t)60, (Double_t)9.0, (Double_t)15.0);
	
	th3 = new TH1F("th3", "Histogram", 60, 9.0, 15.0);
	th4 = new TH1F("th4", "Histogram", 60, 9.0, 15.0);
	
	th1->Eval(tf1);
	th2->Eval(tf2);

	//th1->Draw();
	//th2->Draw("same");
	//priors
	float p1 = 2.0/3.0;
	float p2 = 1.0/3.0;

	//cout << "Beginning first iteration" << endl;
	for (int i = 0; i <= th1->GetNbinsX(); i++) {
		float ev = th1->GetBinContent(i)*p1+th2->GetBinContent(i)*p2;
		th3->SetBinContent(i, (th1->GetBinContent(i)*p1)/ev);
		th4->SetBinContent(i, (th2->GetBinContent(i)*p2)/ev);
		//pw[i] = (w1[i]*p1)+(w2[i]*p2);
		//x2[i] = th1->GetBinCenter(i);
	}
	
	th3->SetLineColor(kBlack);
	th3->Draw("C");
	th4->SetLineColor(kRed);
	th4->Draw("CSAME");
	
	//Problem 3
	//c3 = new TCanvas("c3", "2.3", 800,400);

	//delete w1;
	//delete w2;
}
