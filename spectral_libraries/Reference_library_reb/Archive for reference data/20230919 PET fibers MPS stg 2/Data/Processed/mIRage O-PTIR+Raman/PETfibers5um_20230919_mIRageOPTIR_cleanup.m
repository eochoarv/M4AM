clear all
clc

%% 

Path ='C:\Users\rparham\Documents\Projects\M4AM\CC-OPTIR project\Reference library\20230919 PET fibers MPS stg 2\Data\Processed\mIRage O-PTIR+Raman\firstlookplots\';


%% 

wavenumbersIR = xlsread('20230919_5umPETfibers_stg2MPS10min_cc2.xlsx',1,'A2:A1029');
spectraExp = xlsread('20230919_5umPETfibers_stg2MPS10min_cc2.xlsx',1,'B2:BX1029');

% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR

%% 

for i = 1:size(spectraExp,2)
    figure
    plot(wavenumbersIR,spectraExp(:,i))

    xlabel('Wavenumbers (cm^{-1})')
    ylabel('IR Intensity')
    title(['Particle ',num2str(i)])

    set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
    box on

    FileNamePlot = sprintf('%s%s%d.fig',Path,['Particle ',num2str(i)]);
    FileNamePlot2 = sprintf('%s%s%d.jpg',Path,['Particle ',num2str(i)]);
    saveas(gcf,FileNamePlot,'fig')
    saveas(gcf,FileNamePlot2,'jpg')

    close all
end

%% cleaning up

keep = [13,14,19,36,49,61,65,67];
%best spectra: 36 and 49

spectraExpTrans = transpose(spectraExp);

for j = 1:length(keep)
    for i = 1:size(spectraExp,2)
        if i ~= keep(j)
            spectraExpTrans(i,1) = -9999;
        end
    end
end

spectraExpTrans(spectraExpTrans(:,1) == -9999,:) = [];

spectraExpClean = transpose(spectraExpTrans);
% spectraExpTrans
% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR

%% change nans to zeros

for i = 1:size(spectraExpClean,2)
    spectraExpClean(458:875,i) = 0;
end

%% change negatives to zeros

for i = 1:size(spectraExpClean,2)
    for j = 1:size(spectraExpClean,1)
        if spectraExpClean(j,i) < 0
            spectraExpClean(j,i) = 0;
        end
    end
end

%% 
% 
% for i = 1:4
%     standardsIR{i} = xlsread('all standards avg norm OPTIR spectra',i,'A2:C1029');
% end

%% 


save('PETfibers5um_20230919_mIRageOPTIR_cleandata.mat','spectraExpClean','wavenumbersIR','standardsIR')

