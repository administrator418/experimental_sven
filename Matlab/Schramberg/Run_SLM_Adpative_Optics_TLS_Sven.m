%%  first section
clc; clear; close all;
wd = '\\srvditz1\lac\Studenten\AE_VoE_Stud\JIkai Wang\02 Matlab\09-SLMcontrol\SLM_NEW\SLM_new\';
addpath(wd);
cd(wd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%Create Zernike Coefficients
% num_rows = 21150; % Number of rows is amount of phasemasks
% 
% %%Zernike Coefficients (n and m)
% n_modes = [0  1  1  2  2  2  3  3  4  3  3  4  4  4  4];   % n (Zernike Mode)
% m_modes = [0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4];   % m (Zernike Mode)
% mm = -0.4;
% nn = 0.4; 
% 
% %%Initialize the matrix to hold all rows of weights
% zernike_weights = zeros(num_rows, 15);
% 
% %%Generate weights for each row
% for row = 1:num_rows
%     weights = zeros(1, 15); % Initialize weights
%     weights(1:15) = 0; % First three weights are 0
%     %weights(5:15) = mm + (nn-mm) * rand(1,11); % Remaining weights are random within the range
%     zernike_weights(row, :) = weights; % Store the weights in the matrix
% end
% 
% %%Specify the directory where the file will be saved
% d = '\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Schramberg_Preperation\Datacollection';
% 
% %%Save as CSV file in the specified directory
% csvwrite(fullfile(d, 'zernike_coefficients_schramberg_test.csv'), zernike_weights);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Read in the Zernike Coefficients
T=readtable('\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Schramberg_Preperation\Datacollection\zernike_coefficients_schramberg.csv');
p=T{:,:};
save('coeff.mat','p')
coeff = load("coeff.mat");
coeff=coeff.p;
%Select coeff
coeff = coeff(13765:13774,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  third section
sizeRAD = 3.57/2;             % FU04: X:3.49mm Y:2.65mm  @ second order moment  and using D=3.57@ 90% energy circled.
pidi = 0.0125;                % pixel size of the SLM
%lambda = 1.03E-3;             % wavelength
xx = (-sizeRAD : pidi : sizeRAD);
xxn = xx/sizeRAD;
[XX, YY] = meshgrid(xxn, xxn);
[THETA, RR] = cart2pol(XX,YY);
idx = RR<=1;
%%  fourth section and fitting it into the new SLM
%  generat a phase map for begnning the MPC and laseR, as we can use the
%  grating order and fit it in the new SLM.
posi=[1922 57 1272 1024];
figure1 = figure('Menu','none','ToolBar','none','Position', posi);
axes1 = axes('Parent',figure1,'YDir','reverse',...
    'Position',[0 0 1 1],...
    'PlotBoxAspectRatio',[1272 1024 1],...
    'Layer','top',...
    'DataAspectRatio',[1 1 1])
%imagesc(FIN, [0 255]);axis equal;colormap gray;
axis off;
    n =     [0  1  1  2  2  2  3  3  4  3  3  4  4  4  4];      %n (Zernike Mode)
    m =     [0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4];    %m (Zernike Mode)
    weight = [0  0  0  0 0  0 0  0  0  0  0  0  0  0  0];   %Zernike-coefficients
    sag = zernfun(n,m,RR(idx), THETA(idx));
    stackZER = zeros(length(xx), length(xx), length(n));
    tmp = zeros(length(xx), length(xx));
    WF = zeros(size(tmp));

    for laufp = 1 :length(n)
         tmp(idx) = sag(:, laufp);
         stackZER(:,:,laufp) = tmp;
         WF = WF + pi * weight(laufp) * tmp;   
    end

% fitting into the new SLM
    xxs = (-636 : 635)*pidi;
    yys = (-512 : 511)*pidi;
    [XXs, YYs] = meshgrid(xxs, yys);
    RRs = sqrt(XXs.^2+YYs.^2);
    THETAs = atan2(YYs, XXs);
    A0 = zeros(1024, 1272);                         % SLM size 
    ce = [636 512];                                  % Zentrumskoordinate

% Match of aberration map with SLM + Grating
    A1=zeros(286,493);
    A_1=zeros(286,493);
    WF_new=[A1,WF,A_1];
    A2=zeros(369,1272);
    A_2=zeros(369,1272);
    WF_new_1=[A_2;WF_new;A2];      % creat the rows
    WF_function= exp(1i*WF_new_1);

% display grating and wavefront
    n = 1.5;
    lambda = 1.03E-3;
    alpha =2;
    beta = (n-1)*alpha*pi/180*2*pi/lambda;
   %T= angle(exp(-1i*beta*RRs));
    ell =0;
    per = 0.125;     %  0.125 is fixed
    Kx = 2*pi/per;   %  We use 8 for grating
    Ky = 2*pi/per;
    T_grat = exp(1i*(Kx*XXs+Ky*YYs));
    T = angle(WF_function.*T_grat);
    %imagesc(T);
    [mskSLM, phase] = dispSLM_WFcorr_new(T, lambda, 'on', [1922 57 1272 1024], A0);   % the row and column need to be same and it is a dummy figure
%% fifth section: loop to generate the aberration from 0 to 0.4lambda
% here we should define the range for each coefficients.
%[0,0], 'piston', [1,-1], 'Y_tilt', [1,1], 'X_tilt', [2,-2], 'oblique Asti', [2,0], 'Defocus',[2,2], 'vertical Asti',
%[3,-3],'Vertical trefoil', [3,-1], 'vertical coma', [3,1], horizontal
% coma, [3,3],oblique trefoil,[4,-4],oblique trefoil, [4,-2], oblique
% secondary asti, [4,0], primary speherical,[4,+2], Vertical secondary
% asti, [4,+4] vertical quadrafoil
% figure1 = figure('Menu','none','ToolBar','none','Position', posi);
% axes1 = axes('Parent',figure1,'YDir','reverse',...
%     'Position',[0 0 1 1],...
%     'PlotBoxAspectRatio',[1272 1024 1],...
%     'Layer','top',...
%     'DataAspectRatio',[1 1 1])
% axis off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define start time and create timestamps
    NUMBER_OF_IMAGES =5;
    start_time = "14:11:00";
    start_time_slm = datestr(datetime(start_time, 'InputFormat', 'HH:mm:ss') + seconds(0), 'HH:MM:SS');
    interval_seconds = 300; %4 before
    timestamps = time_stamp(start_time_slm, NUMBER_OF_IMAGES, interval_seconds);


disp('Waiting...')
for z=1:NUMBER_OF_IMAGES
    %%%%%%%%%%%%%%%%%% Waiting until the timestamp is reached%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %tic;
    timestamp = datetime(timestamps{z}, 'InputFormat', 'HH:mm:ss');
    current_time = datetime('now', 'Format', 'HH:mm:ss');

    % Adjust for the case where the target time is on the next day
    if timestamp < current_time
        timestamp = timestamp + caldays(1);
    end

    while current_time < timestamp
        pause(0.1); % sleep for 100ms to avoid busy waiting
        current_time = datetime('now', 'Format', 'HH:mm:ss');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    n =     [0  1  1  2  2  2  3  3  4  3  3  4  4  4  4  ];   %n (Zernike Mode)
    m =     [0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4  ];    %m (Zernike Mode)
    mm = -0.4;
    nn =  0.4; 
    
    %%%%%%%%%Load Weights%%%%%%%
    weight = coeff(z,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    sag = zernfun(n,m,RR(idx), THETA(idx));
    stackZER = zeros(length(xx), length(xx), length(n));
    tmp = zeros(length(xx), length(xx));
    WF = zeros(size(tmp));

    %%%%%%%%%%%%%%%%%%%%%%%%%Create the Phasemask%%%%%%%%%%%%%%%%%%%%%%%%
    for laufp = 1 :length(n)
         tmp(idx) = sag(:, laufp);
         stackZER(:,:,laufp) = tmp;
         WF = WF + pi * weight(laufp) *tmp;          
    end
     xxs = (-636 : 635)*pidi;
     yys = (-512 : 511)*pidi;
     [XXs, YYs] = meshgrid(xxs, yys);
     RRs = sqrt(XXs.^2+YYs.^2);
     THETAs = atan2(YYs, XXs);
     A0 = zeros(1024, 1272);                          % SLM size 
     ce = [636 512];                                  % the center of SLM
    %%%%%%%%%%%%% Match of aberration map with SLM + Grating%%%%%%%%%%%%%%% 
    A1=zeros(286,493);
    A_1=zeros(286,493);
    WF_new=[A1,WF,A_1];
    A2=zeros(369,1272);
    A_2=zeros(369,1272);
    WF_new_1=[A_2;WF_new;A2];        % creat the rows
    WF_function= exp(1i*WF_new_1);
 
    %%%%%%%%%%% Save Phasemask of wavefront %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d = 'C:\Sven\Schramberg_Datacollection\phasemask\';
    P = angle(WF_function);
    Phase_mask= uint8(255*(P+pi)/2/pi);
    current_datetime = datestr(now, 'yyyy-mm-dd HH-MM-SS');
    fileName = [d current_datetime '_' num2str(z) '_0' '.png'];
    imwrite(Phase_mask, fileName);

    %%%%%%%%%%%%% display grating + wavefront%%%%%%%%%%%%%%%%%
    n = 1.5;
    lambda = 1.03E-3;
    alpha =2;
    beta = (n-1)*alpha*pi/180*2*pi/lambda;
    ell =0;                      %  charge
    per = 0.125;                 %  0.125 is fixed
    Kx = 2*pi/per;               %  We use 8 for grating
    Ky = 2*pi/per;
    T_grat = exp(1i*(Kx*XXs+Ky*YYs));
    T = angle(WF_function.*exp(1i*ell*THETAs).*T_grat);
    %imagesc(T);
    
    [mskSLM, phase] = dispSLM_WFcorr_new(T, lambda, 'on', [1922 57 1272 1024],A0);   % the row and column need to be same
    
    pause(1)  
    % This is the pause time for detecting the wavefront, also possible to change for M2 measurment.
    disp(z)                                                   
    
end 
disp('finish')