import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CS540P1 {
    private static final String traingData = "src/mnist_train.csv";
	private static final String[] testFiles = {"src/test_0.csv", "src/test_2.csv"};
	
	//Parameters for predictions
    static String firstId = "0";
    static String secondId = "2";
    
    //Parameters for weights
    private static final int max_num_iterations = 500;
    static double learning_rate = .1;

    
    public static List<List<Double>> prepTrainingFile(String fileLocation) throws FileNotFoundException, IOException {
    	
        List<List<Double>> records = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(fileLocation))) {
            String currLine;
            while (reader.readLine() != null) {
            	currLine = reader.readLine();
                String[] string_values = currLine.split(",");
                if (!string_values[0].equals(firstId) && !string_values[0].contentEquals(secondId)) continue;
                Double[] csvDoubles = new Double[string_values.length];
                
                //Classify 
                if (firstId.equals(string_values[0])){
                	csvDoubles[0] = 0.0;
                }
                else{
                	csvDoubles[0] = 1.0;
                }
                for (int i = 1; i < string_values.length; i++) {
                    csvDoubles[i] = Double.parseDouble(string_values[i])/255.0; // features
                }
                records.add(Arrays.asList(csvDoubles));
            }
        } 
        return records;
    }
    
    public static List<List<Double>> prepTestFiles(String[] fileLocations) throws FileNotFoundException, IOException {
        List<List<Double>> records = new ArrayList<>();
        for (int i = 0; i < fileLocations.length; i++){
        	String file_path = fileLocations[i];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Parse comma delimited CSV
                String[] string_values = line.split(",");
                Double[] double_values = new Double[string_values.length + 1];

                switch(i){
                case 0:
                	double_values[0] = 0.0;
                	break; // label 0
                
                case 1:
                	double_values[0] = 1.0;
                	break;
                }
                
                for (int j = 1; j < string_values.length; j++) {
                    double_values[j] = Double.parseDouble(string_values[j])/255.0; // features
                }
                

                records.add(Arrays.asList(double_values));
            }       
        }
    }
    return records;
    }
    
    public static void main(String[] args) throws IOException {
        // Parse csv files
        List<List<Double>> trainingData = prepTrainingFile(traingData);
        List<List<Double>> testRecords = prepTestFiles(testFiles);
        
        // Print Results and weights
        PrintWriter output = new PrintWriter("src/output.txt");
        PrintWriter weight = new PrintWriter("src/weights.txt");
        
        // Other Variables
        double tempBias = 0;
        
        // Create Arrays
        Double[] weights = new Double[testRecords.size() ];
        Random rand = new Random();
        Double bias = rand.nextDouble();
        
        //Create weights array
        for (int i = 0; i < weights.length; i++){
        	weights[i] = rand.nextDouble();
        }
        
        
        for (int step = 0; ; step++) {
            // Calculate a_i array
            Double[] a = new Double[trainingData.size()];
            for (int i = 0; i < trainingData.size(); i++) {
                double sumW = 0;
                
                for (int j = 0; j < weights.length; j++) {
                    sumW += weights[j] * trainingData.get(i).get(j+1);
                }
                a[i] = 1.0 / (1 + Math.exp(-1 * (sumW + bias)));
            }

            // Update weights and bias
            for (int j = 0; j < weights.length; j++) {
                double tempWeight = 0;
                for (int i = 0; i < trainingData.size(); i++) {
                    tempWeight += (a[i] - trainingData.get(i).get(0)) * trainingData.get(i).get(j+1);
                }
                //update and output new weight
                weights[j] = weights[j] - learning_rate * tempWeight;            
            }

            // Update bias
            for (int i = 0; i < trainingData.size(); i++) {
                tempBias += (a[i] - trainingData.get(i).get(0));
            }
            bias -= learning_rate * tempBias;
            // Print Bias
            weight.println(bias);
        
            //Cost
            Double currCost = 0.0;
            Double potentialCost = 0.0;
            
            currCost = potentialCost;
            potentialCost = 0.0;
            for (int i = 0; i < trainingData.size(); i++) {
                if (trainingData.get(i).get(0) == 0.0) {
                    if (a[i] > 0.9999) potentialCost += 100.0; // something large
                    else potentialCost -= Math.log(1 - a[i]);
                }
                else if (trainingData.get(i).get(0) == 1.0) {
                    if (a[i] < 0.0001) potentialCost += 100.0;
                    else potentialCost -= Math.log(a[i]);
                }
            }

            // Check for convergence
            if (Math.abs(potentialCost - currCost) < 0.0001) break;
            if (step > max_num_iterations) {
                System.out.println("Max iterations reached!");
                break;
            }
            System.out.println(step);
        }
        
        // See how we did
        double correctGuess = 0;
        for (int i = 0; i < testRecords.size(); i++) {
            double sum_wx = 0;
            for (int j = 0; j < weights.length; j++) {
                sum_wx += weights[j] * testRecords.get(i).get(j+1);
            }
            double a = 1.0 / (1 + Math.exp(-1 * (sum_wx + bias)));
            if (a < 0.5 && testRecords.get(i).get(0) == 0.0){
            	correctGuess++;
            }
            if (a >= 0.5 && testRecords.get(i).get(0) == 1.0){
            	correctGuess++;
            }
        
            if (a < 0.5){ 
            	System.out.print("Predicted: " + firstId);
            	 if(testRecords.get(i).get(0) == 0){
 	            	System.out.println(", Actual: " + firstId);
 	            	output.println(firstId);
 	            }
 	            else{
 	            	System.out.println(", Actual: " + secondId);
 	            	output.println(firstId);
 	            }
            }
            if (a >= 0.5){ 
	            System.out.print("Predicted: " + secondId);
	            if(testRecords.get(i).get(0) == 0){
	            	System.out.println(", Actual: " + firstId);
	            	output.println(firstId);
	            }
	            else{
	            	System.out.println(", Actual: " + secondId);
	            	output.println(secondId);
	            }
            }
        }
        
        System.out.println("Prediction Accuracy: " + correctGuess/testRecords.size());
        output.close();
        weight.close();
    }
}
