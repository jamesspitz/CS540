import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class NeuralNetwork {
	
    private static final String DIR_PATH = "./FEI";
    private static final String WRITE_TO_PATH = "./output.csv";
    private static final String PATH_TO_TEST = "./test.csv";
    private static final Double ALPHA = 0.0;
    private static final int MAX_EPOCHS = 0;
    private static final double EPSILON = 0;
    
    // Recommended resizing
    static int height = 36;
    static int width = 26;
    
    // Initialize weight matrices & bias vectors & costs
    static Double prev_c = 0.0;
    static Double curr_c = 0.0;
    static Random rng = new Random();
    static int m = height * width; // hard-coded
    static Double[][] w_1 = new Double[m][m];
    static Double[] w_2 = new Double[m];
    static Double[] b_1 = new Double[m];
    static Double b_2 = rng.nextDouble() - rng.nextDouble();
    static Double[] a_1 = new Double[m];
    static Double a_2 = null;

	

	public static void main(String[] args) throws IOException {
		
		Preprocessor pp = new Preprocessor(height, width);
        pp.convert_to_cvs(DIR_PATH, WRITE_TO_PATH);
        
        // Parse csv files
        List<List<Double>> records = prepFiles(WRITE_TO_PATH);
        List<List<Double>> test_records = prepFiles(PATH_TO_TEST);
        
        for (int i = 0; i < m; i++) {
            w_2[i] = rng.nextDouble() - rng.nextDouble();
            b_1[i] = rng.nextDouble() - rng.nextDouble();
            for (int j = 0; j < w_1.length; j++) w_1[i][j] = rng.nextDouble() - rng.nextDouble();
        }
            
        for (int num_epochs = 1; ; num_epochs++) {
            // Alternative to the random permutation by uniformly picking an integer between 0 and n-1 from the training set
            // Note that you might want to try Knuth/Fisher-Yates shuffles approaches
            rng = new Random(); // init empty seed each epochs for randomness
            /*int randomArray = records.size();
            shuffleFY(records);*/
            
            // Creates an index array for the training set and use Fisher-Yates to shuffle the indices
            int[] randomArray = null;
            for(int i = 0; i < records.size() -1; i++) {
            	randomArray[i] = i;
            }
            shuffleFY(randomArray);
            
            
            
            for (int recItr = 0; recItr < records.size(); recItr++) { // each epoch has # of iterations equals # training instances
                
            	// Calculate a_1 & a_2 of the current instance
            	int i = recItr;
                ForwardProp(records, randomArray[i]);

                // Back propagation
                Double dcDb2 = 0.0; // TODO: update dc_db2; currently hard-code it to zero
                Double[] dc_dw2j = new Double[m];
                
                for (int j = 0; j < dc_dw2j.length; j++) {
                    dc_dw2j[j] = 0.0; // TODO: update dc_dw2j array; currently hard-code it to zero
                }   
                
                Double[] dc_db1j = new Double[m];
                
                for (int j = 0; j < dc_db1j.length; j++) {
                    dc_db1j[j] = 0.0; // TODO: update dc_db1j array; currently hard-code it to zero
                }   
                Double[][] dc_dw1j = new Double[m][m];
                
                for (int j = 0; j < m; j++) { // hard-coded size m
                    for (int j_prime = 0; j_prime < m; j_prime++) {
                        dc_dw1j[j][j_prime] = 0.0; // TODO: update dc_dw1j 2D array; currently hard-code it to zero
                    }
                }

                // Update weights & biases
                for (int j = 0; j < m; j++) {
                    w_2[j] -= ALPHA * 0; // TODO: update w_2 array; currently hard-code it to zero
                    b_1[j] -= ALPHA * 0; // TODO: update b_1 array; currently hard-code it to zero
                    for (int j_prime = 0; j_prime < m; j_prime++) {
                        w_1[j][j_prime] -= ALPHA * 0; // TODO: update w_1 2D array; currently hard-code it to zero
                    }
                }
                b_2 -= ALPHA * 0; // TODO: update b_2; currently hard-code it to zero
            }   
            
            // Calculate cost function & inference after each epoch (just to show progress)
            prev_c = curr_c;
            curr_c = 0.0;
            for (int i = 0; i < records.size(); i++) {
                ForwardProp(records, i);
                curr_c += Math.pow((records.get(i).get(0) - a_2), 2);           
            }
            curr_c = 0.5 * curr_c;
            
            double num_correct = 0;
            for (int i = 0; i < test_records.size(); i++) {
                ForwardProp(test_records, i);

                if (a_2 >= 0.5 && test_records.get(i).get(0) == 1.0) num_correct++;
                else if (a_2 < 0.5 && test_records.get(i).get(0) == 0.0) num_correct++;
            }
            System.out.println("Epoch #" + num_epochs + "=> C: " + curr_c + " Accuracy: " + num_correct/test_records.size());           
            
            // Check for convergence
            if (Math.abs(curr_c - prev_c) < EPSILON) break;
            else if (num_epochs >= MAX_EPOCHS) { // termination condition
                System.out.println("Reached the maximum number of num_epochs. "
                        + "Maybe try a different learning rate?");
                break;
            }
        }

        // Inference step
        double num_correct = 0;
        for (int i = 0; i < test_records.size(); i++) {
            ForwardProp(test_records, i);

            if (a_2 >= 0.5 && test_records.get(i).get(0) == 1.0) num_correct++;
            else if (a_2 < 0.5 && test_records.get(i).get(0) == 0.0) num_correct++;

            if (a_2 < 0.5) System.out.print("Predicted: 0, ");
            else System.out.print("Predicted: 1, ");
            System.out.println("Actual: " + test_records.get(i).get(0) + " a_2: " + a_2);
        }
        System.out.println("Accuracy: " + num_correct/test_records.size());
    }
    
    /* Calculate a_i arrays */
    private static void ForwardProp(List<List<Double>> records, int curr_index) {
        for (int i = 0; i < m; i++) {
            double sumWx = 0;
            for (int j = 0; j < m; j++) {
                sumWx += w_1[i][j] * records.get(curr_index).get(j+1);
            }
            a_1[i] = 0.0; // TODO: update a_1 array; currently hard-code it to zero
        }

        double sumAw = 0;
        for (int j = 0; j < m; j++) {
            sumAw += a_1[j] * w_2[j];
        }
        a_2 = 0.0; // TODO: update a_2 array; currently hard-code it to zero
    }
    
    // Process CSV
    public static List<List<Double>> prepFiles(String file_path) throws FileNotFoundException, IOException {
        List<List<Double>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] string_values = line.split(",");
                Double[] double_values = new Double[string_values.length];
                double_values[0] = Double.parseDouble(string_values[0]); // label
                for (int i = 1; i < string_values.length; i++) {
                    double_values[i] = Double.parseDouble(string_values[i]); // features
                }
                records.add(Arrays.asList(double_values));
            }
        }
        return records;
    }
    
    // Fisher-Yates Algo
    public static int[] shuffleFY(int[] intArray) {
    	Random rand = new Random();
        for (int i = intArray.length - 1; i > 0; i--)
        {
        	// Swap
			int index = rand.nextInt(i + 1);
			int num = intArray[index];
			intArray[index] = intArray[i];
			intArray[i] = num;       
        }
        return intArray;
      }
    
    
    
    
    
}
