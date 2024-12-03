%% old SLM to collect the two images
clc; clear; close all;
%%
wd = '\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Matlab\SLM_old';
addpath(wd);
cd(wd);
%%
sizeRAD = 4.98; %5.98
%sizeRAD = 6;
pidi = 0.02;
lambda = 1.03E-3;
coeff=zeros(20000,15);
xx = (-sizeRAD : pidi : sizeRAD);
xxn = xx/sizeRAD;
[XX, YY] = meshgrid(xxn, xxn);
[THETA, RR] = cart2pol(XX,YY);
idx = RR<=1;
%% phase mask figure
posi=[1922 480 800 600];
figure1 = figure('Menu','none','ToolBar','none','Position', posi);
axes1 = axes('Parent',figure1,'YDir','reverse',...
    'Position',[0 0 1 1],...
    'PlotBoxAspectRatio',[1272 1024 1],...
    'Layer','top',...
    'DataAspectRatio',[1 1 1]);

% Start datacollection.
% command = 'python Sven_Camera2.py';
% system(command);

% Settting8
NUMBER_OF_IMAGES = 200;
start_time = "15:54:00";
start_time_slm = datestr(datetime(start_time, 'InputFormat', 'HH:mm:ss') + seconds(0), 'HH:MM:SS');
interval_seconds = 4;
timestamps = time_stamp(start_time_slm, NUMBER_OF_IMAGES, interval_seconds);


for z = 1:NUMBER_OF_IMAGES
    %tic;
    timestamp = timestamps(z);
%     fprintf('Timestamp: %s\n', timestamp);
% 
     current_time = datestr(datetime(), 'HH:MM:SS');
%     fprintf('Current_Time: %s\n', current_time)

    while datenum(current_time, 'HH:MM:SS') <= datenum(timestamp, 'HH:MM:SS')
        current_time = datestr(now, 'HH:MM:SS');
    end

%     while True
%         current_time = datestr(now, 'HH:MM:SS');
%         %disp(['Current time: ', current_time]); 
%         if strcmp(current_time, timestamp)
%             break;
%         end
%     end


%     posi=[1922 480 800 600];
%     figure1 = figure('Menu','none','ToolBar','none','Position', posi);
%     axes1 = axes('Parent',figure1,'YDir','reverse',...
%     'Position',[0 0 1 1],...
%     'PlotBoxAspectRatio',[1272 1024 1],...
%     'Layer','top',...
%     'DataAspectRatio',[1 1 1]);

    n =  [ 0  1  1  2  2  2  3  3  4  3  3  4  4  4  4  ];   %n (Zernike Mode)
    m =  [ 0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4  ];   %m (Zernike Mode)
    mm=-0.8;   % 0.15 lambda  [0.4 to 0.8 change] %Strenght of the Abreations
    nn= 0.8;    % 0.15 lambda [0.4 to 0.8 change]

    weight(1:3)=0;
    weight(4:15) = mm+(nn-mm)*rand(1,12);   % generate the weightbase from -0.3 lambda to 0.3 lambda
    % Carefull changes same weights
    % display(weight)
    % weight(4:15) = [0.1808    0.0684    0.3618   -0.1114   -0.3270   -0.0105   -0.2675   -0.1941   -0.3986    0.2106    0.2110   -0.3875]
    % weight(4:15) = [0.1675    0.2037   -0.1792    0.1438    0.1241   -0.2699   -0.3048   -0.0013    0.3678   -0.1277    0.0682   -0.2210]
    coeff(z,:)=weight;
    %class = class(coeff);
    %disp(['Datatype: ' class]);
    
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
    %figure;
    %imagesc(WF); colorbar; % abberation 599*599
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
    %% running the python environment
    %% pyversion C:\Users\01_TLD712_photron\AppData\Local\Programs\Python\Python39\python.exe
    
    %% saving the phase mask of wavefront
    %d='\\srvditz1\\lac\\Studenten\\AE_VoE_Stud\\Sven Burckhard\\Experimental_data\\phasemask\\';
    %d= '\\srvditz1\HOME$\01_TLD712_photron\Desktop\Sven_Burckhard\Phasemask\';
    d = 'C:\Local_Scripts\phasemask\';
    P = angle(WF_function);
    Phase_mask= uint8(255*(P+pi)/2/pi);
    current_datetime = datestr(now, 'yyyy-mm-dd HH-MM-SS');
    fileName = [d current_datetime '_' num2str(z) '_0' '.png'];
    imwrite(Phase_mask, fileName);

    %imwrite(Phase_mask, [d num2str(z) 'phase_mask.png']);,
    
    % display grating + wavefront
    n = 1.5;
    lambda = 1.03E-3;
    alpha =2;
    beta = (n-1)*alpha*pi/180*2*pi/lambda;
    ell =0;                      %  charge
%   per = 0.125;                 %  0.125 is fixed
%   Kx = 2*pi/per;               %  We use 8 for grating
%   Ky = 2*pi/per;
    Kx=75;
    Ky=0;
    T_grat = exp(1i*(Kx*XXs+Ky*YYs));
    T = angle(WF_function.*exp(1i*ell*THETAs).*T_grat);
    [mskSLM, phase] = dispSLM_WFcorr(T, lambda, 'on', [1922 480 800 600], A0);
    
    % collect the near and far field and save it in the local computer
    pause(1)    % 10sec                  % This is the pause time for detecting the wavefront, also possible to change for M2 measurment
    %pyrunfile("Copy_of_Camera2.py") % use the python file for data collection:  camera ID1
    
    % Definiere den Pfad zum Speicherort der CSV-Datei
%         csvFolderPath = '\\srvditz1\HOME$\01_TLD712_photron\Desktop\Sven_Burckhard\';
%         csvFileName = 'execution_time.csv';
%         csvFilePath = fullfile(csvFolderPath, csvFileName);
%         tic;
    %pyrunfile("Copy_of_Camera2.py") % use the python file for data collection : camera ID2 
%         elapsedTime = toc;
    
     % Speichere die Zeit in einer CSV-Datei
%         csvFileName = 'execution_time.csv';
%         header = {'PythonFile', 'ExecutionTime'};
%         data = {z, elapsedTime};
%         writecell(data, '\\srvditz1\HOME$\01_TLD712_photron\Desktop\Sven_Burckhard\time_stamps.csv',  'WriteMode', 'append');

        %elapsedTime = toc;
        %disp(elapsedTime);
        z;
        disp(z);
end

%path_excel = 'E:\\coeff_vortex';
writematrix(coeff, 'C:\Local_Scripts\coeff_evaluation.xlsx');
%writematrix(coeff, path_excel);
         
disp('Finish')
                              