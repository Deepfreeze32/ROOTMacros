//The first ROOT assignment
//Pattern Recognition
// Travis Cramer

void Cramer_mp2() {
	float x1[] = {9, 10, 10.7, 11, 11.5, 12, 13, 14, 15}; 
	float y1[] = {0, 0.08, 0.24, .21, 0.15, .18, .30, .17, .05}; 
 	float y2[] = {0.03, .14, .27, .33, .36, .3, .18, .04, 0}; 
	
	//PROBLEM 1
 	c1 = new TCanvas("c1", "2.1", 800, 400);
 	//c1->Divide(2,1);

 	//First case
	tg1 = new TGraph(9,x1,y1);
	tg1->SetMarkerColor(kBlack);
	tg1->SetMarkerStyle(20);

	//Second case
	tg2 = new TGraph(9,x1,y2);
	tg2->SetMarkerColor(kRed);
	tg2->SetMarkerStyle(20);

	tf1 = new TF1("tf1","gaus(0) + gaus(3)", 9, 15);
	tf1->SetParNames("Amp", "Mean", "Width", "Amp", "Mean", "Width");
	tf1->SetParameters(.2, 10.5, 0.5, 0.38, 13.01, 0.5);

	tf2 = new TF1("tf2", "gaus(0) + gaus(3)", 9, 15);
	tf2->SetParNames("Amp", "Mean", "Width", "Amp", "Mean", "Width");
	tf2->SetParameters(0.21,10.8,0.75,0.3,13,3);

	//Draw the data
	tf1->SetLineColor(kBlack);
	tf2->SetLineColor(kRed);
	tf2->Draw();
	tf1->Draw("same");
	tg1->Draw("P");
	tg2->Draw("P");
	tg1->Fit(tf1);
	tg2->Fit(tf2);
	
	//Set title for stuff
	tf2->SetTitle(".2,10.5,0.5,0.38,13.01,0.5,0.21,10.8,0.75,0.3,13,3");

	//PROBLEM 2
	c2 = new TCanvas("c2", "2.2", 800,400);

	th1 = new TH1F("th1", "Histogram", 60, 9.0, 15.0);
	th2 = new TH1F("th2", "Histogram", 60, 9.0, 15.0);
	
	th3 = new TH1F("th3", "Histogram", 60, 9.0, 15.0);
	th4 = new TH1F("th4", "Histogram", 60, 9.0, 15.0);
	
	//Popualte bins
	th1->Eval(tf1);
	th2->Eval(tf2);

	//priors
	float p1 = 2.0/3.0;
	float p2 = 1.0/3.0;

	for (int i = 0; i <= th1->GetNbinsX(); i++) {
		//Calculate Evidence
		float ev = th1->GetBinContent(i)*p1+th2->GetBinContent(i)*p2;
		//Set content
		th3->SetBinContent(i, (th1->GetBinContent(i)*p1)/ev);
		th4->SetBinContent(i, (th2->GetBinContent(i)*p2)/ev);
	}
	
	//Draw
	th3->SetLineColor(kBlack);
	th3->Draw("C");
	th4->SetLineColor(kRed);
	th4->Draw("CSAME");
	
	//Problem 3
	c3 = new TCanvas("c3", "2.3", 800,400);

	th5 = new TH1F("th5","Histogram",60,9.0,15.0);
	th6 = new TH1F("th6","Line Hist",60, 9.0, 15.0);

	for (int i = 0; i <= th1->GetNbinsX(); i++) {
		th5->SetBinContent(i,(th1->GetBinContent(i)/th2->GetBinContent(i)));
		th6->SetBinContent(i, p2/p1);
	}

	//Draw our stuff!
	th6->SetLineColor(kRed);
	th6->SetLineStyle(7);
	th6->Draw("C");
	th5->Draw("CSAME");
}
