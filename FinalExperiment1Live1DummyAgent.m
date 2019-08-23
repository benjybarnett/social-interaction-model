%set values
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;
noWeights=9;
N=3;


agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);
agent1Locations = [];
allFitnessScoresOneDummy=[];
fitnessScoresOneDummy = [];
dummyAgent = recordedAgent;
dummyInitActiv = recordedAgentInitActiv;
for t = 1:NoTrials
    initialConditionsMatrix = randn(N,1);
    
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
    agent1StartPoint =0;
    agentOneLocation = zeros(1,length(time));
    agentOneLocation(:,1) = agent1StartPoint;
    

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

        %agent velcoity
        agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
        agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
        agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;

        %agent location
        
        agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);

        %input is distance to other agent mapped between 1 and 0 only
        %when agents are within 0 - 200 units of space to eachother
        %on-off sensing essentially
        distanceToOther = -(abs(dummyAgent(:,i) - agentOneLocation(:,i)));
        if(distanceToOther > -200 && distanceToOther< 0)
            bla =[bla distanceToOther];
            norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
            I(:,:) = norm_data(end)*sensorGain;
        else
            I(:,:) = 0;
        end

        if((agentOneLocation(:,i) < dummyAgent(:,i) +(20)) && (agentOneLocation(:,i)>dummyAgent(:,i)-(20)))
            crossLocations = [crossLocations agentOneLocation(:,i)];
        end

     end


    if(isempty(crossLocations))
        fitness= 0;
    else
        fitness = abs(crossLocations(end));

    end
    fitnessScoresOneDummy(t)= fitness;
    allFitnessScoresOneDummy=[allFitnessScoresOneDummy fitness];
    agent1Locations=[agent1Locations; agentOneLocation];
    
end

meanFitnessOneDummy = mean(fitnessScoresOneDummy);
medianFitnessOneDummy = median(fitnessScoresOneDummy);
stdDevOneDummy = std(fitnessScoresOneDummy);

[highestScoringTrial, indxHighestScoringTrial]=max(allFitnessScoresTwoLive);
highestAgent1 = agent1Locations(indxHighestScoringTrial,:);

TF = isoutlier(fitnessScoresOneDummy);
count = length(fitnessScoresOneDummy);
while(count>0)
    
    if(TF(:,count) == 1)
        fitnessScoresOneDummy(:,count) = [];
        
        count =count- 1;
    else
        count =count- 1;
    end
end


meanFitnessOneDummyOutliersRemoved = mean(fitnessScoresOneDummy);
medianFitnessOneDummyOutliersRemoved = median(fitnessScoresOneDummy);
stdDevOneDummyOutliersRemoved = std(fitnessScoresOneDummy);
iqrangeOneDummy = iqr(fitnessScoresOneDummy);
stdErrOneDummy = std(fitnessScoresOneDummy)/sqrt(length(fitnessScoresOneDummy));


subplot(1,2,2)
plot(time,highestAgent1)
hold on
plot(time,dummyAgent)
hold off
legend('Agent One', 'Dummy Agent');
xlabel('Time');
ylabel('Location');
function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end