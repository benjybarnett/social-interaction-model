%set values
NoPositions = 101;
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;


startPoints = linspace(-50,50,101);
noWeights=9;

displacementAndFitness = zeros(length(startPoints),5);

agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);

dummyLocation = recordedAgent;
for p = 1:length(startPoints)
    fitnessScores = [];
    
    for t = 1:NoTrials
        initialConditionsMatrix = randn(N,1);
        initialConditionsMatrix2 = randn(N,1);
        
        noiseSD = sqrt(0.5);
        noiseMean = 0;
        
        noise3 = noiseSD.*randn(1,length(time)) + noiseMean;
        noise4 = noiseSD.*randn(1,length(time)) + noiseMean;
        bla = linspace(-200,0,1000);
        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditionsMatrix(:,:);
        
        
        %set start locations
        agent1StartPoint = startPoints(:,p);
        agentOneLocation = zeros(1,length(time));
        agentOneLocation(:,1) = agent1StartPoint;
        %dummylocation already preset
        
        %agent input
        I1=0;
        I = zeros(N,1);
        I(:,:) = I1; 

        %array storing locations where the agents cross
        crossLocations = [];
        
        for i= 2:length(time)


            if(rand<0.3)
                I=0;
            end

            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ I(:,:)));

            %agent Velocity
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;

            %agent location
            
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);

            %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceToOther = -(abs(dummyLocation(:,i) - agentOneLocation(:,i)));
            if(distanceToOther > -200 && distanceToOther< 0)
                bla =[bla distanceToOther];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                I(:,:) = norm_data(end)*sensorGain;
            else
                I(:,:) = 0;
            end

            if((agentOneLocation(:,i) < dummyLocation(:,i) +(20)) && (agentOneLocation(:,i)>dummyLocation(:,i)-(20)))
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
    displacementAndFitness(p,1) = agent1StartPoint - dummyLocation(:,1);%store displacement
    displacementAndFitness(p,2) = meanFitness; % store fitness
    displacementAndFitness(p,3) = stdDev;
    displacementAndFitness(p,4) = stdErr;
    displacementAndFitness(p,5) = iqrange;
end
meanFitnessOneDummy = displacementAndFitness(:,2);
sortedDataDummy = sortrows(displacementAndFitness);
subplot(1,3,3)
errorbar(sortedDataDummy(:,1),sortedDataDummy(:,2),sortedDataDummy(:,3))
title('Dummy Agent');
xlabel('Relative Displacement')
ylabel('Mean Fitness');
meanFitnessDummyOverall = mean(displacementAndFitness(:,2));
ylim(yl);



function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end