%set values
NoPositions = 101;
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;


startPoints = linspace(-50,50,101);
startPoints2=linspace(-25,25,101);
noWeights=9;

displacementAndFitness = zeros(length(startPoints),5);
displacementAndFitnessLiveAgents = zeros(length(startPoints),3);

agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);

dummyAgent = recordedAgent;
for p = 1:length(startPoints)
    fitnessScores = [];
    fitnessScoresLiveAgents = [];
    noiseSD = sqrt(0.5);
    noiseMean = 0;
    noiseMatrix = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix2 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix3 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix4 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix5 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix6 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    initialConditionsMatrix = randn(N,NoTrials);
    initialConditionsMatrix2 = randn(N,NoTrials);
    
    for t = 1:NoTrials
        noise=noiseMatrix(t,:);
        noise2=noiseMatrix2(t,:);
        noise3=noiseMatrix3(t,:);
        noise4=noiseMatrix4(t,:);
        noise5=noiseMatrix5(t,:);
        noise6=noiseMatrix6(t,:);
        initialConditions = initialConditionsMatrix(:,t);
        initialConditions2 = initialConditionsMatrix2(:,t);
        
        bla = linspace(-200,0,1000);
        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditions(:,:);
        %agent3 
        agentThree = zeros(N,length(time));
        agentThree(:,1) = initialConditions3(:,:);
        
        %set start locations
        agent1StartPoint = startPoints(:,p);
        agentOneLocation = zeros(1,length(time));
        agentOneLocation(:,1) = agent1StartPoint;
        %dummylocation already preset
        agent3StartPoint = startPoints2(:,p);
        agentThreeLocation(:,1) = agent3StartPoint;
        
        
        
         %agent  input
        agentOneI = zeros(N,1);
        
        agentThreeI = zeros(N,1); 

        %array storing locations where the agents cross
        crossLocations = [];
        crossLocationsLiveAgents=[];
        for i= 2:length(time)
            
            if(rand<0.3)
                agentOneI(:,:)=0;
                
                agentThreeI(:,:)=0;
            end
            
            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ agentOneI(:,:)));
            
            %integrated equation of CTRNN agent three
            agentThree(:,i) = agentThree(:,i-1) +dt*(-agentThree(:,i-1)+tanh(agentWeights*agentThree(:,i-1)+ agentThreeI(:,:)));
            
            %agent Velocity
            
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;
            agentThreeVelocityLeft = (agentThree(2,i) + noise5(:,i))*outputGain;
            agentThreeVelocityRight = (agentThree(3,i) + noise6(:,i))*outputGain;
            agentThreeVelocity = (agentThreeVelocityLeft-agentThreeVelocityRight);

            %agent location
            
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);
            agentThreeLocation(:,i) = agentThreeLocation(:,i-1)+(agentThreeVelocity);

            %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceBetweenAgentOneDummy = -(abs(dummyAgent(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentOneThree = -(abs(agentThreeLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentdummyThree = -(abs(agentThreeLocation(:,i) - dummyAgent(:,i)));
            if(distanceBetweenAgentOneDummy > -200 && distanceBetweenAgentOneDummy< 0)
                bla =[bla distanceBetweenAgentOneDummy];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(1,:) = norm_data(end)*sensorGain;
                
            else
                agentOneI(1,:) = 0;
                
            end
            if(distanceBetweenAgentOneThree > -200 && distanceBetweenAgentOneThree< 0)
                bla =[bla distanceBetweenAgentOneThree];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(2,:) = norm_data(end)*sensorGain;
                agentThreeI(1,:)= norm_data(end)*sensorGain;
            else
                agentOneI(2,:) = 0;
                agentThreeI(1,:)=0;
            end
            if(distanceBetweenAgentdummyThree > -200 && distanceBetweenAgentdummyThree< 0)
                bla =[bla distanceBetweenAgentdummyThree];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
               
                agentThreeI(2,:)= norm_data(end)*sensorGain;
            else
               
                agentThreeI(2,:)=0;
            end
            
            agentLocations = [agentOneLocation(:,i),dummyAgent(:,i),agentThreeLocation(:,i)];
            if(range(agentLocations)<40)
                crossLocations = [crossLocations agentOneLocation(:,i)];
            end
            if((agentOneLocation(:,i) < agentThreeLocation(:,i) +(20)) && (agentOneLocation(:,i)>agentThreeLocation(:,i)-(20)))
                crossLocationsLiveAgents = [crossLocationsLiveAgents agentOneLocation(:,i)];
            end
                    
    
        end
         
         
        if(isempty(crossLocations))
            fitness=0;
        else
            fitness = abs(crossLocations(end));
        end
        fitnessScores(t)= fitness;
        if(isempty(crossLocationsLiveAgents))
            fitnessLiveAgents = 0;
        else
            fitnessLiveAgents = abs(crossLocationsLiveAgents(end));
        end
        fitnessScoresLiveAgents(t) = fitnessLiveAgents;
    end
    
    meanFitness = mean(fitnessScores);
    meanFitnessLiveAgents = mean(fitnessScoresLiveAgents);
    stdDev = std(fitnessScores);
    stdDevLiveAgents = std(fitnessScoresLiveAgents);
    iqrange = iqr(fitnessScores);
    stdErr = std(fitnessScores)/sqrt(length(fitnessScores));
    displacementAndFitness(p,1) = agent1StartPoint-agent2StartPoint;%store displacement
    displacementAndFitnessLiveAgents(p,1) = agent1StartPoint-agent2StartPoint;
    displacementAndFitness(p,2) = meanFitness; % store fitness
    displacementAndFitnessLiveAgents(p,2) = meanFitnessLiveAgents;
    displacementAndFitness(p,3) = stdDev;
    displacementAndFitnessLiveAgents(p,3) = stdDevLiveAgents;
    displacementAndFitness(p,4) = stdErr;
    displacementAndFitness(p,5) = iqrange;
end
meanFitnessOneDummyTwoLive = displacementAndFitness(:,2);
sortedDataDummyTwoLive = sortrows(displacementAndFitness);
subplot(1,4,3)
errorbar(sortedDataDummyTwoLive(:,1),sortedDataDummyTwoLive(:,2),sortedDataDummyTwoLive(:,3))
title('With One Dummy Agent');
xlabel('Relative Displacement')
ylabel('Mean Fitness');
%xlim([min(startPoints2),max(startPoints2)])
sortedDataTwoLiveOnly = sortrows(displacementAndFitnessLiveAgents);
subplot(1,4,4)
errorbar(sortedDataTwoLiveOnly(:,1),sortedDataTwoLiveOnly(:,2),sortedDataTwoLiveOnly(:,3));
title('The Two Live Agents (Dummy Still Present)')
xlabel('Relative Displacement From Dummy')
ylabel('Mean Fitness');
%xlim([min(startPoints2),max(startPoints2)])
meanFitnessDummyOverall = mean(displacementAndFitness(:,2));
meanFitnessTwoLiveOverall=mean(displacementAndFitnessLiveAgents(p,2));
%ylim(yl);



function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end