import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class CS540P1 {

	// Specify hyper-parameters HERE
    private static final String Training_Dest = "./mnist_train.csv";
    private static final String Testing_Path = "./mnist_test.csv";
    private static final String Comma = ",";
    private static final int max_num_iterations = 100;
    static double learning_rate = 0.1;
    static String first_digit = "1";
    static String second_digit = "9";
    
    public static List<List<Double>> parseRecords(String file_path) throws FileNotFoundException, IOException {
        List<List<Double>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] string_values = line.split(Comma);
                if (!string_values[0].equals(first_digit) && !string_values[0].contentEquals(second_digit)) continue;
                Double[] double_values = new Double[string_values.length];
                if (first_digit.equals(string_values[0])) double_values[0] = 0.0; // label 0
                else double_values[0] = 1.0; // label 1
                for (int i = 1; i < string_values.length; i++) {
                    double_values[i] = Double.parseDouble(string_values[i])/255.0; // features
                }
                records.add(Arrays.asList(double_values));
            }
        }
        return records;
    }
    
    public static void main(String[] args) throws IOException {
        // Parse csv files
        List<List<Double>> records = parseRecords(Training_Dest);
        List<List<Double>> test_records = parseRecords(Testing_Path);
        
        // Initialize b & c & array w
        Random rng = new Random();
        Double b = rng.nextDouble(); // init bias
        Double[] w = new Double[784]; // hard-coded 784
        for (int i = 0; i < w.length; i++) w[i] = rng.nextDouble(); // init weights
        Double prev_c = 0.0;
        Double curr_c = 0.0;
        
        // Gradient descent step
        for (int iteration = 0; ; iteration++) {
            // Calculate a_i array
            Double[] a = new Double[records.size()];
            for (int i = 0; i < records.size(); i++) {
                double sum_wx = 0;
                for (int j = 0; j < w.length; j++) {
                    sum_wx += w[j] * records.get(i).get(j+1);
                }
                a[i] = 1.0 / (1 + Math.exp(-1 * (sum_wx + b)));
            }

            // Update weights and bias
            for (int j = 0; j < w.length; j++) {
                double w_temp = 0;
                for (int i = 0; i < records.size(); i++) {
                    w_temp += (a[i] - records.get(i).get(0)) * records.get(i).get(j+1);
                }
                w[j] = w[j] - learning_rate * w_temp; // update weights             
            }

            // Update bias
            double b_temp = 0;
            for (int i = 0; i < records.size(); i++) {
                b_temp += (a[i] - records.get(i).get(0));
            }
            b -= learning_rate * b_temp;
        
            // Calculate cost function
            prev_c = curr_c;
            curr_c = 0.0;
            for (int i = 0; i < records.size(); i++) {
                if (records.get(i).get(0) == 0.0) {
                    if (a[i] > 0.9999) curr_c += 100.0; // something large
                    else curr_c -= Math.log(1 - a[i]);
                }
                else if (records.get(i).get(0) == 1.0) {
                    if (a[i] < 0.0001) curr_c += 100.0;
                    else curr_c -= Math.log(a[i]);
                }
            }

            // Check for convergence
            if (Math.abs(curr_c - prev_c) < 0.0001) break;
            else if (iteration > max_num_iterations) { // termination condition
                System.out.println("Reached the maximum number of iterations. "
                        + "Maybe try a different learning rate?");
                break;
            }
            System.out.println(iteration);
        }
        
        // Inference step
        double num_correct = 0;
        for (int i = 0; i < test_records.size(); i++) {
            double sum_wx = 0;
            for (int j = 0; j < w.length; j++) {
                sum_wx += w[j] * test_records.get(i).get(j+1);
            }
            double a = 1.0 / (1 + Math.exp(-1 * (sum_wx + b)));
            if (a < 0.5 && test_records.get(i).get(0) == 0.0) num_correct++;
            else if (a >= 0.5 && test_records.get(i).get(0) == 1.0) num_correct++;
        
            if (a < 0.5) System.out.print("Predicted: 0, ");
            else System.out.print("Predicted: 1, ");
            System.out.println("Actual: " + test_records.get(i).get(0));
        }
        
        System.out.println("Accuracy: " + num_correct/test_records.size());
    }

}
