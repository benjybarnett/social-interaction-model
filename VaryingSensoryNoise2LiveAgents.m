%set values
noNoiseLevels = 50;
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;


noiseLevels = linspace(0.05,0.5,noNoiseLevels);
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
        noiseSD = sqrt(0.5);
        noiseMean = 0;
        noise = noiseSD.*randn(1,length(time)) + noiseMean;
        noise2 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise3 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise4 = noiseSD.*randn(1,length(time)) + noiseMean;
        bla = linspace(-200,0,1000);
        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditionsMatrix(:,:);
        
        %agent two
        agentTwo = zeros(N,length(time));
        agentTwo(:,1) = initialConditionsMatrix2(:,:);
        %set start locations
        agent1StartPoint = 0;
        agentOneLocation = zeros(1,length(time));
        agentOneLocation(:,1) = agent1StartPoint;
        agentTwoLocation = zeros(1,length(time));
        agent2StartPoint = 0; %agent 2 always starst at 0
        agentTwoLocation(:,1) = agent2StartPoint;
        
        %agent input
        I1=0;
        I = zeros(N,1);
        I(:,:) = I1; 

        %array storing locations where the agents cross
        crossLocations = [];
        
        for i= 2:length(time)
            %sensory noise
            if(rand<noiseLevels(p))
                I = not(I);
            end
               

            %integrated equation of CTRNN dummy agent
            agentTwo(:,i) =agentTwo(:,i-1) +dt*(-agentTwo(:,i-1)+tanh(agentWeights*agentTwo(:,i-1)+ I));

            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ I));

            %agent Velocity
            agentTwoVelocityLeft = (agentTwo(2,i) + noise(:,i))*outputGain;
            agentTwoVelocityRight = (agentTwo(3,i) + noise2(:,i))*outputGain;
            agentTwoVelocity = agentTwoVelocityLeft-agentTwoVelocityRight;
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;

            %agent location
            agentTwoLocation(:,i) = agentTwoLocation(:,i-1) + (agentTwoVelocity);
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);

            

             %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceToOther = -(abs(agentTwoLocation(:,i) - agentOneLocation(:,i)));
            if(distanceToOther > -200 && distanceToOther< 0)
                bla =[bla distanceToOther];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                I(:,:) = norm_data(end)*sensorGain;
            else
                I(:,:) = 0;
            end

            if((agentOneLocation(:,i) < agentTwoLocation(:,i) +(20)) && (agentOneLocation(:,i)>agentTwoLocation(:,i)-(20)))
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
    sensorNoiseLevelAndFitness(p,1) = noiseLevels(p);%store noiuse level
    sensorNoiseLevelAndFitness(p,2) = meanFitness; % store fitness
    sensorNoiseLevelAndFitness(p,3) = stdDev;
    sensorNoiseLevelAndFitness(p,4)= stdErr;
    sensorNoiseLevelAndFitness(p,5) = iqrange;
end


hold off
subplot(1,2,2)
errorbar(sensorNoiseLevelAndFitness(:,1),sensorNoiseLevelAndFitness(:,2),sensorNoiseLevelAndFitness(:,3))
title('Manipulating Sensor Noise');
xlabel('Sensor Noise Level')
ylabel('Mean Fitness')
xlim([0.05,0.5])




function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end