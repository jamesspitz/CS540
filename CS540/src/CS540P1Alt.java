import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
	
public class CS540P1Alt {
	
	// File locations and parsing symbol
	private static String trainFile = "./mnist_train.csv";
	private static String[] testFiles = {"./test_0.csv", "./test_2.csv"};
	
	// Regression Parameters
	double learningRate = .2;
	static int maxItr = 50;
	
	// Define values
	static String firstId = "1";
	static String secondId = "9";
	
	public static List<List<Double>> prepareFiles(String FileLocation) throws FileNotFoundException, IOException {
		
		// Create 2D array for doubles
		List<List<Double>> csvArray = new ArrayList<>();
		
		// Iterate through the file array
		for( int i = 0; i < testFiles.length; i++){
			File csvFile = new File(testFiles[i]);
			
			// Declare Scanner
			Scanner csvInput;
			
			String[]inputValues = null;
			Double[] csvDoubles = {0.0};
			
						
			try{
				csvInput = new Scanner(csvFile);
				
				// Read in lines
				while(csvInput.hasNextLine()){
					String currLine = csvInput.next();
					// Parse comma delimited CSV
					inputValues = currLine.split(",");
					}
				
				// Iterate and classify values
				for( int j = 0; j< inputValues.length; j++){
					if (!inputValues[0].equals(firstId) && !inputValues[0].equals(secondId)){
						continue;
					}
					if (firstId.equals(inputValues[0])){
						csvDoubles[0] = (double) 0;
					}
					else csvDoubles[0] = (double) 1;
				}
				
				// Build double array from string array
				for (int k = 1; k < inputValues.length; k++) {
					// Divide the input value by RGB value 255
					csvDoubles[k] = Double.parseDouble(inputValues[k])/255.0;
                }
				
				// Add to 2D array
				csvArray.add(Arrays.asList(csvDoubles));
						
			// Error Handling  
			}catch (FileNotFoundException e){
				if(testFiles.length == 0){
					System.out.println("No files selected, please select a file!");
					}
				if(testFiles.length == 1){
				System.out.println("Cannot find file, check the directory location and try again!");
				}
				else{
					System.out.println("Cannot find files, check the directory location and try again!");
				}
			}
		}	
		return csvArray;
	}
	
	 public static void main(String[] args) throws IOException {
	        
		 	// Prep CSV
	        List<List<Double>> trainingData = prepareFiles(trainFile);
	        List<List<Double>> test_records = prepareFiles(Testing_Path);
}
