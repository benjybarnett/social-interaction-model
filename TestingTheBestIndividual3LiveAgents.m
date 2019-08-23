%set values
NoPositions = 101;
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;


startPoints = linspace(-50,50,101);
%startPoints2=linspace(-25,25,101);
noWeights=9;

displacementAndFitness = zeros(length(startPoints),5);

agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);
agent2Locations = [];
allFitnessScores=[];
for p = 1:length(startPoints)
    fitnessScores = [];
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
    initialConditionsMatrix3 = randn(N,NoTrials);

    
    for t = 1:NoTrials
        noise=noiseMatrix(t,:);
        noise2=noiseMatrix2(t,:);
        noise3=noiseMatrix3(t,:);
        noise4=noiseMatrix4(t,:);
        noise5=noiseMatrix5(t,:);
        noise6=noiseMatrix6(t,:);
        initialConditions = initialConditionsMatrix(:,t);
        initialConditions2 = initialConditionsMatrix2(:,t);
        initialConditions3 = initialConditionsMatrix3(:,t);

        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditions(:,:);
        %agent2
        agentTwo = zeros(N,length(time));
        agentTwo(:,1) = initialConditions2(:,:); 
        %agent Three
        agentThree = zeros(N,length(time));
        agentThree(:,1) = initialConditions3(:,:);
        
        bla = linspace(-200,0,1000);
       
        
        
        %set start locations
        agent1StartPoint = startPoints(:,p);
        agentOneLocation = zeros(1,length(time));
        agentOneLocation(:,1) = agent1StartPoint;
        agentTwoLocation = zeros(1,length(time));
        agent2StartPoint = 0; %agent 2 always starst at 0
        agentTwoLocation(:,1) = agent2StartPoint;
        agentThreeLocation = zeros(1,length(time));
        %agent3StartPoint = startPoints2(:,p);
        agentThreeLocation(:,1) = agent1StartPoint;%starts with agent one
        
        %agent  input
        agentOneI = zeros(N,1);
        agentTwoI = zeros(N,1);
        agentThreeI = zeros(N,1);
    

        %array storing locations where the agents cross
        crossLocations = [];
        
        for i= 2:length(time)
            
            if(rand<0.3)
                agentOneI(:,:)=0;
                agentTwoI(:,:)=0;
                agentThreeI(:,:)=0;
            end
            %integrated equation of CTRNN  agent two
            agentTwo(:,i) =agentTwo(:,i-1) +dt*(-agentTwo(:,i-1)+tanh(agentWeights*agentTwo(:,i-1)+ agentTwoI(:,:)));

            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ agentOneI(:,:)));
            
            %integrated equation of CTRNN agent three
            agentThree(:,i) = agentThree(:,i-1) +dt*(-agentThree(:,i-1)+tanh(agentWeights*agentThree(:,i-1)+ agentThreeI(:,:)));
            
            %agent Velocity
            agentTwoVelocityLeft = (agentTwo(2,i) + noise(:,i))*outputGain;
            agentTwoVelocityRight = (agentTwo(3,i) + noise2(:,i))*outputGain;
            agentTwoVelocity = (agentTwoVelocityLeft-agentTwoVelocityRight);%negative as facing other direction
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;
            agentThreeVelocityLeft = (agentThree(2,i) + noise5(:,i))*outputGain;
            agentThreeVelocityRight = (agentThree(3,i) + noise6(:,i))*outputGain;
            agentThreeVelocity = (agentThreeVelocityLeft-agentThreeVelocityRight);

            %agent location
            agentTwoLocation(:,i) = agentTwoLocation(:,i-1) +(agentTwoVelocity);
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);
            agentThreeLocation(:,i) = agentThreeLocation(:,i-1)+(agentThreeVelocity);

            %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceBetweenAgentOneTwo = -(abs(agentTwoLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentOneThree = -(abs(agentThreeLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentTwoThree = -(abs(agentThreeLocation(:,i) - agentTwoLocation(:,i)));
            if(distanceBetweenAgentOneTwo > -200 && distanceBetweenAgentOneTwo< 0)
                bla =[bla distanceBetweenAgentOneTwo];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(1,:) = norm_data(end)*sensorGain;
                agentTwoI(1,:)= norm_data(end)*sensorGain;
            else
                agentOneI(1,:) = 0;
                agentTwoI(1,:)=0;
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
            if(distanceBetweenAgentTwoThree > -200 && distanceBetweenAgentTwoThree< 0)
                bla =[bla distanceBetweenAgentTwoThree];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentTwoI(2,:) = norm_data(end)*sensorGain;
                agentThreeI(2,:)= norm_data(end)*sensorGain;
            else
                agentTwoI(2,:) = 0;
                agentThreeI(2,:)=0;
            end
            
            agentLocations = [agentOneLocation(:,i),agentTwoLocation(:,i),agentThreeLocation(:,i)];
            if(range(agentLocations)<40)
                crossLocations = [crossLocations agentOneLocation(:,i)];
            end

         
         
            
           

            
        end
    if(isempty(crossLocations))
        fitness= 0;
    else
        fitness = abs(crossLocations(end));

    end
    fitnessScores(t)= fitness;
    allFitnessScores=[allFitnessScores fitness];
    agent2Locations=[agent2Locations; agentTwoLocation];
    
    end
    meanFitness = mean(fitnessScores);
    stdDev = std(fitnessScores);
    iqrange = iqr(fitnessScores);
    stdErr = std(fitnessScores)/sqrt(length(fitnessScores));
    displacementAndFitness(p,1) = agent1StartPoint-agent2StartPoint;%range from agent 1 and agent 3 to agent 2
    displacementAndFitness(p,2) = meanFitness; % store fitness
    displacementAndFitness(p,3) = stdDev;
    displacementAndFitness(p,4)= stdErr;
    displacementAndFitness(p,5) = iqrange;
end
[highestScoringTrial, indxHighestScoringTrial]=max(allFitnessScores);
recordedAgent = agent2Locations(indxHighestScoringTrial,:);
meanFitnessThreeLiveAgents = displacementAndFitness(:,2);
sortedDataLiveAgents = sortrows(displacementAndFitness);
subplot(1,4,1)
plot(bestFitness)
xlim([0,NoGenerations])
xlabel('Number of Generations')
ylabel('Best Fitness in Population')
subplot(1,4,2)
errorbar(sortedDataLiveAgents(:,1),sortedDataLiveAgents(:,2),sortedDataLiveAgents(:,3))
title('Three Live Agents');
xlabel('Relative Displacement')
ylabel('Mean Fitness')
%xlim([min(startPoints2),max(startPoints2)])
meanFitnessLiveAgentsOverall = mean(displacementAndFitness(:,2));
yl=ylim;


function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end