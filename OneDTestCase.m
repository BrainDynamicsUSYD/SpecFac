% This script demonstrates the 1D test case in 
% Brain Dynamics and Structure-Function Relationships via Spectral Factorization and the Transfer Function
% J. A. Henderson, M. Dhamala and P. A. Robinson, 2020
%
% Created by J. A. Henderson, July 2020

clear all

% Generate 1D grid with periodic boundaries
Npts=50; % number of points in the 1D line
circum=15e-2; % circumference in m
rad=circum/(2*pi); % circle radius (periodic boundaries) 
dtheta=2*pi/Npts; % angular resolution
dX=rad*dtheta; % spatial resolution
theta=0:dtheta:2*pi-dtheta; % anglular coordinate of each point
X=rad*theta; % spatial coordinate of each point

% Wrap angles to implement periodic boundaries
for i=1:Npts
    for j=1:Npts
        angles1(i,j)=wrapTo2Pi(theta(j)-theta(i));
        angles2(i,j)=wrapTo2Pi(theta(i)-theta(j));
    end
end
r1=rad*angles1;
r2=rad*angles2;

r=min(r1,r2); % find minimum distance between points (moving either direction around circle)

G=ones(Npts,Npts); % set gains to one

% Neural Field Theory Parameters
alpha=100; % synaptic decay rate [s^-1]
beta=350; % synaptic rise rate [s^-1]
r_e=84e-3; % axonal range [m]
vel=9; % conduction speed [m/s]
gamma_e=vel/r_e; % decay rate [s^-1]

fmax=0.5*vel/(circum/Npts); % maximum frequency
Deltaf=3.8; % frequency sampling resolution

f=0:Deltaf:fmax; % generate +ve frequencies starting from 0
f=[fliplr(-f(2:end)),f(1:end)]; % add -ve frequencies;
fmax=max(f); % change fmax if fmax is not an integer multiple of Deltaf
fposind=find(f>=0); % indexes to +ve frequencies
fnegind=find(f<0); % indexes to -ve frequencies

eta=-0.5; % degree of asymmetery in the 1D propagator, range -1 to 1

mev=0; % initialise max eigenvalue to zero
% Now construct Gamma and Lambda
for fi=1:length(f)
    
    for i=1:Npts
        for j=1:Npts
            L=(1+sqrt(-1)*2*pi*f(fi)/alpha)^-1*(1+sqrt(-1)*2*pi*f(fi)/beta)^-1;
            q=(1+sqrt(-1)*2*pi*f(fi))/gamma_e;
            
            if i~=j % no self propagtion
                % compute the 1D propagator
                Gammatemp=dX*0.5/vel*( (1-eta)*exp( (-sqrt(-1)*2*pi*f(fi)-gamma_e)*(r1(j,i)/vel))+ (1+eta)*exp( (-sqrt(-1)*2*pi*f(fi)-gamma_e)*(r2(j,i)/vel)) );
                
            else
                Gammatemp=0;
            end
            
            Lambda(j,i,fi)=G(j,i)*Gammatemp;
            Gamma(j,i,fi)=G(j,i)*Gammatemp;
        end
    end
    mev(fi)=max(real(eig(Lambda(:,:,fi)))); % save maximum eigenvalue
end

Lambda=0.85*Lambda/max(real(mev)); % scale so max eigenvalue is 0.85 (must be less than one for stability)
Gamma=0.85*Gamma/max(real(mev)); % scale so max eigenvalue is 0.85 (must be less than one for stability)

%% Now construct transfer function
T=[];
for i=1:length(f)
    T(:,:,i)=inv(eye(Npts)-Lambda(:,:,i));
end
%% Now construct the spectral density matrix (frequency domain correlation matrix)
S=[];
for i=1:length(f)
    S(:,:,i)=T(:,:,i)*(T(:,:,i)');
end
%% Now do spectral factorization to recover the transfer function

[T_est,Z] = specfactorization_wilson(S(:,:,fposind),fmax); % Input positive frequencies to the Wilson algorithm
T_est=cat(3,flip(conj(T_est(:,:,2:end)),3),T_est); % add negative freqency components to T
epsilon=sqrt(sum(sum(sum(abs(T-T_est).^2)))/sum(sum(sum(abs(T).^2)))) % compute the error between the actual and estimated transfer functions
    
%% Now compute estimated Lambda and Gamma
for i=1:size(T_est,3)
    Lambda_est(:,:,i)=eye(Npts)-inv(T_est(:,:,i));
    L=(1+sqrt(-1)*2*pi*f(i)/alpha)^-1*(1+sqrt(-1)*2*pi*f(i)/beta)^-1;
    Gamma_est(:,:,i)=Lambda_est(:,:,i)/L;
end
%% Now inverse Fourier transform to obtain temporal quantities

Lambdatemp=2*fmax*ifft(ifftshift(Lambda,3),[],3,'symmetric');
Gammatemp=2*fmax*ifft(Gamma,[],3,'symmetric');
Ttemp=2*fmax*ifft(ifftshift(T,3),[],3,'symmetric');

Ttemp_est=2*fmax*ifft(ifftshift(T_est,3),[],3,'symmetric');
Lambdatemp_est=2*fmax*ifft(ifftshift(Lambda_est,3),[],3,'symmetric');
Gammatemp_est=2*fmax*ifft(Gamma_est,[],3,'symmetric');

%% Setup the figure for plotting
figno=1;
figure(figno)
clf

lw=1; % line width for plots
fsz=10; % font size

%% Plot T in space at some points in time
subplot(3,2,1)
hold on

t=(0:(length(f)-1))/(2*fmax); % time points

N=zeros(size(X))'; % input vector
N(round(length(X)/2))=1; % only provide input at the central point

cols=['r','b','g','c','m','k','y']; % some plotting line colors

ti=4:6:24; % choose timepoints to plot
tplot=t(ti);

for tpi=1:length(ti)
    Q=Ttemp(:,:,ti(tpi))*N; % actual system activity
    p(tpi)=plot(X,Q,'LineWidth',lw,'Color',cols(tpi));
end

for tpi=1:length(ti)
    Q=Ttemp_est(:,:,ti(tpi))*N; % estimated system activity
end

xlabel('$X$ (m)','interpreter','latex')
ylabel('$T(X,\tau)$','interpreter','latex')
title('(a)')
box on
ylim([-2,85])
set(gca,'TickLength',[0.04, 0.01])

% create a legend to show times foe each line in the plot
for i=1:length(tplot)
    legstr{i}=['$\tau=~$',num2str(round(tplot(i)*1e4)/10),'$~{\rm ms}$'];
end

lgd=legend(legstr); 
set(lgd,'box','off')
set(lgd,'interpreter','latex')
set(gca,'FontSize',fsz)
%% Plot Lambda in space at some points in time
subplot(3,2,2)
hold on

for tpi=1:length(tplot)
    Q=Lambdatemp(:,:,ti(tpi))*N; % actual system activity
    p(tpi)=plot(X,Q,'LineWidth',lw,'Color',cols(tpi));
end

for tpi=1:length(tplot)
    Q=Lambdatemp_est(:,:,ti(tpi))*N; % estimated system activity
end

xlabel('$X$ (m)','interpreter','latex')
ylabel('$\Lambda(X,\tau)$','interpreter','latex')
title('(b)')
box on
set(gca,'TickLength',[0.04, 0.01])

% create a legend to show times foe each line in the plot
lgd=legend(legstr);
set(lgd,'box','off')
set(lgd,'interpreter','latex')
set(gca,'FontSize',fsz)

%% Plot one element of T in time
subplot(3,2,3)

N=zeros(size(X))'; % input vector
N(1)=1; % input to first point only

[sep,point]=min(abs(X-0.05)); % choose the point about 5cm away

for ti=1:length(t)
    Qtemp=Ttemp(:,:,ti)*N;
    Q(ti)=Qtemp(point); % actual
    
    Qtemp=Ttemp_est(:,:,ti)*N;
    Q_est(ti)=Qtemp(point); % estimated
end
plot(t,Q,'k')
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$T(\tau)$','interpreter','latex')
title('(c)')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
xlim([0,0.015])
ylim([-1,65])
%% Plot difference between T and T_est
subplot(3,2,5)
Q=zeros(1,length(t));
Q_est=zeros(1,length(t));

for ti=1:length(t)
    Qtemp=Ttemp(:,:,ti)*N;
    Q(ti)=Qtemp(point); % actual
    
    Qtemp=Ttemp_est(:,:,ti)*N;
    Q_est(ti)=Qtemp(point); % estimated
end
plot(t,Q-Q_est,'k')
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$T(\tau)-T_{est}(\tau)$','interpreter','latex')
title('(e)')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
xlim([0,max(t)])
ylim([-2.2,2.2])
%% Plot one element of Lambda in time
subplot(3,2,4)

Q=zeros(1,length(t));
Q_est=zeros(1,length(t));

for ti=1:length(t)
    Qtemp=Lambdatemp(:,:,ti)*N;
    Q(ti)=Qtemp(point);
end
plot(t,Q,'k')
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Lambda(\tau)$','interpreter','latex')
title('(d)')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
xlim([0,0.015])
%% Plot difference between Lambda and Lambda_est
subplot(3,2,6)

Q=zeros(1,length(t));
Q_est=zeros(1,length(t));

for ti=1:length(t)
    Qtemp=Lambdatemp(:,:,ti)*N;
    Q(ti)=Qtemp(point);
    
    Qtemp=Lambdatemp_est(:,:,ti)*N;
    Q_est(ti)=Qtemp(point);
end
plot(t,Q-Q_est,'k')

xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Lambda(\tau)-\Lambda_{est}(\tau)$','interpreter','latex')
title('(f)')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
xlim([0,max(t)])
