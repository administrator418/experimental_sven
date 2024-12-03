%% old SLM to collect the two images
clc; clear; close all;
wd = '\\srvditz1\lac\Studenten\AE_VoE_Stud\JIkai Wang\02 Matlab\09-SLMcontrol\SLM_NEW\SLM_new\';
addpath(wd);
cd(wd);
NUMBER_OF_IMAGES = 200;

%%
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
%generat a phase map for begnning the MPC and laseR, as we can use the
%grating order and fit it in the new SLM.
% posi=[1922 57 1272 1024];
% figure1 = figure('Menu','none','ToolBar','none','Position', posi);
% axes1 = axes('Parent',figure1,'YDir','reverse',...
%     'Position',[0 0 1 1],...
%     'PlotBoxAspectRatio',[1272 1024 1],...
%     'Layer','top',...
%     'DataAspectRatio',[1 1 1])
%imagesc(FIN, [0 255]);axis equal;colormap gray;
%axis off;
    n =     [0  1  1  2  2  2  3  3  4  3  3  4  4  4  4];      %n (Zernike Mode)
    m =     [0  1 -1  0  2 -2  1 -1  0  3 -3  2 -2  4 -4];    %m (Zernike Mode)
    %weight = [0  0  0  0 -0.02545  -0.0686 -0.1172 -0.1132  0.1854  -0.1732  -0.1901  0.0898  0.2130  -0.1682  0.0476];   %Zernike-coefficients
    weight = [0  0  0  0  0  0  0  0  0  0   0  0   0  0  0];   %Zernike-coefficients
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
    [mskSLM, phase] = dispSLM_WFcorr_new(T, lambda, 'on', [1922 57 1272 1024], A0); 
%%
%pyenv(Version= "C:\Users\01_TLD712_photron\AppData\Local\Programs\Python\Python39-32\python.exe");
NUMBER_OF_IMAGES = 1;
start_time = "16:53:00";
start_time_slm = datestr(datetime(start_time, 'InputFormat', 'HH:mm:ss') + seconds(0), 'HH:MM:SS');
interval_seconds = 180;
timestamps = time_stamp(start_time_slm, NUMBER_OF_IMAGES, interval_seconds);

for z = 1:NUMBER_OF_IMAGES
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
         
    %% Predict the Phasemask for the SLM
     xxs = (-636 : 635)*pidi;
     yys = (-512 : 511)*pidi;
     [XXs, YYs] = meshgrid(xxs, yys);
     RRs = sqrt(XXs.^2+YYs.^2);
     THETAs = atan2(YYs, XXs);
     A0 = zeros(1024, 1272);                          % SLM size 
     ce = [636 512];                                  % the center of SLM


   

    %Load the compensate phasemasks
    % Define the folder where the images are located
    %folder = '\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\Evaluation_Vortex\Test_Random_200_phasemasks\phasemask_unwrapped_compensated';
    %folder = '\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\Evaluation_Gaussian\compensatet_phasemask',
    %folder = 'C:\Sven\Schramberg_Datacollection\Evaluation\phasemask_compensated';
    folder = 'C:\Sven\Schramberg_Datacollection\compansated_phasemask_test';

    % Construct the file name for the current image
    fileName = sprintf('%d_idx_compensated_phase.png', z);
    
    % Construct the full file path for the current image
    compensate_phasemask_path = fullfile(folder, fileName);
    
    % Check if the file exists before trying to read it
    if isfile(compensate_phasemask_path)
        % Read the current image
        compensate_phasemask = imread(compensate_phasemask_path);

        
        % Display the image name for verification
        disp(['Processing image: ', fileName]);
    else
        % Display a warning if the file does not exist
        disp(['File not found: ', compensate_phasemask_path]);
    end
    
    % Display the Phasemask
    % Phasemasks are from 0 to 255
    com_phase=double(compensate_phasemask)./255.*2.*pi-pi;
    T_com=exp(1i*com_phase);
    % put the phasemask on the slm 
%     posi=[1922 480 800 600];
%     figure1 = figure('Menu','none','ToolBar','none','Position', posi);
%     axes1 = axes('Parent',figure1,'YDir','reverse',...
%         'Position',[0 0 1 1],...
%         'PlotBoxAspectRatio',[1272 1024 1],...
%         'Layer','top',...
%         'DataAspectRatio',[1 1 1]);
   
    %imagesc(compensate_phasemask, [0 255]);axis equal;colormap gray; % 0 to 1 or 0 to 255

    % display grating + wavefront                    
    per = 0.125;                 %  0.125 is fixed
    Kx = 2*pi/per;               %  We use 8 for grating
    Ky = 2*pi/per;
    T_grat = exp(1i*(Kx*XXs+Ky*YYs));
    T = angle(T_grat.*T_com);
    [mskSLM, phase] = dispSLM_WFcorr_new(T, lambda, 'on', [1922 57 1272 1024],A0);
    
%     % collect images nf and ff
%     pause(2) % This is the pause time for detecting the wavefront, also possible to change for M2 measurment
%     pyrunfile("Camera1.py") % use the python file for data collection :  camera ID1 
%     pyrunfile("Camera2.py") % use the python file for data collection : camera ID2 

    disp(z);



end

disp('Finish')
                              