%%
clc; clear; close all;
%% 
%%
wd = '\\srvditz1\lac\Studenten\AE_VoE_Stud\JIkai Wang\02 Matlab\09-SLMcontrol\SLM_old\';
addpath(wd);
cd(wd);
%%
sizeRAD = 4.98; %5.98
%sizeRAD = 6;
pidi = 0.02;
lambda = 1.03E-3;

xx = (-sizeRAD : pidi : sizeRAD);
xxn = xx/sizeRAD;
[XX, YY] = meshgrid(xxn, xxn);
[THETA, RR] = cart2pol(XX,YY);
idx = RR<=1;

n =         [ 0  1  1  2  2  2  3  3  4  3  3  4  4  4  4  5  5  5  5  5  5 ];   %n (Zernike Mode)
m =         [ 0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4  1 -1  3 -3  5  1 ];   %m (Zernike Mode)

%weight = 2.*[ 0  0  0  0  0.3  -0.14  0.12  -0.2  0.03  0.13  -0.03  0.02  -0.01  0  0  0  0  0  0  0  0 ];   %Zernike-coefficients
%weight = [0	0	0	0.229162604313204	0.442649962679468	-0.111945516221814	0.193698444151891	-0.353480544771367	-0.0351235529446663	-0.296413455927790	0.523471846930019	0.389058025263232	0	-0.690775849518597	0.0514184614908073 ];   %Zernike-coefficients
%ref =    -2.*[ 0  0  0  -1.63  0.0  -0.02  0.05  -0.05  0.0  -0.07  0.02  0.0  -0.00  0  0  0  0  0  0  0  0 ];

%weight = weight + ref;
sag =  zernfun(n,m,RR(idx), THETA(idx));
stackZER = zeros(length(xx), length(xx), length(n));
tmp = zeros(length(xx), length(xx));
WF = zeros(size(tmp));

%figure1 = figure('Color', [1 1 1], 'Position', [100 100 1600 800]);
for laufp = 1 :length(n)
    tmp(idx) = sag(:, laufp);
    stackZER(:,:,laufp) = tmp;
    WF = WF + pi * weight(laufp)*tmp;
    %WF = WF + (2*pi)./(lambda.*1000)*weight(laufp)*tmp; %no *2, because it's reality not simulation
    
end


WF = angle(exp(1i*WF));
figure;
imagesc(WF); colorbar; % abberation 599*599


xxs = (-400 : 399)*pidi;
yys = (-300 : 299)*pidi;
[XXs, YYs] = meshgrid(xxs, yys);
RRs = sqrt(XXs.^2+YYs.^2);
THETAs = atan2(YYs, XXs);
A0 = zeros(600, 800);
ce = [300 400];     % Zentrumskoordinate
% %% create the match of abberation map with SLM
A1=zeros(499,151);
WF=[WF,A1];
A_1=zeros(499,150);
WF=[A_1,WF];
A2=zeros(50,800);
A_2=zeros(51,800);
WF=[A_2;WF;A2];
WF_function= exp(1i*WF);

Kx = 0; %5
Ky = 0;
T_grat = exp(1i*(Kx*XXs+Ky*YYs));
T = angle(WF_function.*T_grat);
imagesc(T);
%T=angle(WF_function);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);
%%   Axicon 
n = 1.5;
lambda = 1.03E-3;
alpha =0.8;     %0.8
beta = (n-1)*alpha*pi/180*2*pi/lambda;
T = angle(exp(-1i*beta*RRs));
%T = exp(-1i*beta*RRs);
%ell = 1;
%Kx = 30;
%T = angle(exp(1i*ell*THETAs).*exp(1i*Kx*XXs));
%T = angle(exp(1i*ell*THETAs));
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);

%%  vortex
n = 1.5;
lambda = 1.03E-3;
alpha =2;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
%T = angle(exp(-1i*beta*RRs));
ell = 2;
Kx = 50;
T = angle(exp(1i*ell*THETAs).*exp(1i*Kx*XXs));
%imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);

%% grating
n = 1.5;
lambda = 1.03E-3;
alpha =2;
%beta = (n-1)*alpha*pi/180*2*pi/lambda;
%T = angle(exp(-1i*beta*RRs));
Kx = 50; %5
T = angle(exp(1i*Kx*XXs));
imagesc(T);
%colormap("gray")
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);

%% Axicon+grating
n = 1.5;
lambda = 1.03E-3;
alpha =0.3;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
T = angle(exp(-1i*beta*RRs));
%T = exp(-1i*beta*RRs);
ell = 1;
Kx = 50;
T = angle(exp(1i*Kx*XXs).*exp(-1i*beta*RRs));
%T = angle(exp(1i*ell*THETAs));
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);


%% Vortex+grating % [//Sven for both vortex and gaussian.
n = 1.5;
lambda = 1.03E-3;
alpha =2;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
%T = angle(exp(-1i*beta*RRs));
ell =0;
Kx = 75;
T = angle(exp(1i*ell*THETAs).*exp(1i*Kx*XXs));
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);

%% axicon with abberation added 
n = 1.5;
lambda = 1.03E-3;
alpha =1;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
T = angle(exp(-1i*beta*RRs).*exp(1i*WF));  % for abberation using multiplex
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);
%%  vortex abberation added
n = 1.5;
lambda = 1.03E-3;
alpha =2;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
ell = 1;
Kx = 60;
T = angle(exp(1i*ell*THETAs).*exp(1i*WF));
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1950 600 800 600], A0);

%% Vortex+ grating
n = 1.5;
lambda = 1.03E-3;
alpha =4;
beta = (n-1)*alpha*pi/180*2*pi/lambda;
%T = angle(exp(-1i*beta*RRs));
ell =2;
Kx = 60;
T = angle(exp(1i*ell*THETAs).*exp(1i*Kx*XXs).*exp(1i*WF));
imagesc(T);
[mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 600 800 600], A0);

%% Simulate illumination

ILL = normGH(xxs, yys, 0, 0, 2.5);

ILL = ILL.*exp(1i*phase);

iF = fftshift(fft2(ifftshift(ILL)));

figure;
subplot(1,2,1);imagesc(xxs,yys,abs(iF).^2);axis equal;colorbar;
subplot(1,2,2);imagesc(xxs,yys,angle(iF));axis equal;colorbar;