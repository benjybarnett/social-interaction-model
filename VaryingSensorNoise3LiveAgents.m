%set values
noNoiseLevels = 50;
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;
N=3;

noiseLevels = linspace(0.05,0.8,noNoiseLevels);
noWeights=9;

sensorNoiseLevelAndFitness = zeros(length(noNoiseLevels),5);

agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);
agent2Locations = [];
allFitnessScores=[];
fitness=[];
for p = 1:length(noiseLevels)
    fitnessScores = [];
    
    for t = 1:NoTrials
        initialConditionsMatrix = randn(N,1);
        initialConditionsMatrix2 = randn(N,1);
        initialConditionsMatrix3 = randn(N,1);
        noiseSD = sqrt(0.5);
        noiseMean = 0;
        noise = noiseSD.*randn(1,length(time)) + noiseMean;
        noise2 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise3 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise4 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise5 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise6 = noiseSD.*randn(1,length(time)) + noiseMean;
        bla = linspace(-200,0,1000);
        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditionsMatrix(:,:);
        
        %agent two
        agentTwo = zeros(N,length(time));
        agentTwo(:,1) = initialConditionsMatrix2(:,:);
        %agent Three
        agentThree = zeros(N,length(time));
        agentThree(:,1) = initialConditionsMatrix3(:,:);
        %set start locations
        agent1StartPoint = 0;
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
            
            if(rand<noiseLevels(p))
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
            fitness=0;
        else
            fitness = abs(crossLocations(end));
        end
        fitnessScores(t)= fitness;
        
    end
    
    meanFitness = mean(fitnessScores);
    stdDev = std(fitnessScores);
    iqrange = iqr(fitnessScores);
    stdErr = std(fitnessScores)/sqrt(length(fitnessScores));
    sensorNoiseLevelAndFitness(p,1) = noiseLevels(p);%store noise levl
    sensorNoiseLevelAndFitness(p,2) = meanFitness; % store fitness
    sensorNoiseLevelAndFitness(p,3) = stdDev;
    sensorNoiseLevelAndFitness(p,4)= stdErr;
    sensorNoiseLevelAndFitness(p,5) = iqrange;
end



subplot(1,2,2)
errorbar(sensorNoiseLevelAndFitness(:,1),sensorNoiseLevelAndFitness(:,2),sensorNoiseLevelAndFitness(:,3))
title('Manipulating Sensor Noise');
xlabel('Sensor Noise Level')
ylabel('Mean Fitness')
xlim([0.05,0.8])
yl=ylim;




function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end