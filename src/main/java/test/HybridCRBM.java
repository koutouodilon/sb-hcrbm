package test;

import org.jblas.DoubleMatrix;
import sbhcrbm.DrugTargetCRBM;
import sbhcrbm.DrugTargetCRBMTool;
import utils.CSVUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class HybridCRBM {
    public static double[][][] X_train_drug_rbm;

    public static DoubleMatrix X_train_drug_rbm_direct;
    public static DoubleMatrix X_train_drug_rbm_indirect;
    public static DoubleMatrix X_train_drug_rbm_isMissing;

    public static double[][][] probDrug;
    public static DoubleMatrix probDrug_direct;
    public static DoubleMatrix probDrug_indirect;

    public static double[][][] X_train_target_rbm;

    public static DoubleMatrix X_train_target_rbm_direct;
    public static DoubleMatrix X_train_target_rbm_indirect;
    public static DoubleMatrix X_train_target_rbm_isMissing;

    public static double[][][] probTarget;
    public static DoubleMatrix probTarget_direct;
    public static DoubleMatrix probTarget_indirect;

    public static void main(String[] args) {

        if (args == null) {
            System.out.println("Required arguments not provided! Exiting...");
            System.exit(-1);
        }

        int n_splits = Integer.parseInt(args[0]);
        String data_folder = args[1];
        String monitoring_folder = args[2];
        String backup_folder = args[3];
        String results_folder = args[4];
        int hidUnits = Integer.parseInt(args[5]);
        double lr = Double.parseDouble(args[6]);
        double weightCost = Double.parseDouble(args[7]);
        int epochs = Integer.parseInt(args[8]);
        double momentum = Double.parseDouble(args[9]);
        int method = Integer.parseInt(args[10]);
        String drug_rbm_folder = args[11];
        String target_rbm_folder = args[12];
        int numDrugs = Integer.parseInt(args[13]);
        int numTargets = Integer.parseInt(args[14]);
        int numClass = Integer.parseInt(args[15]);

        double alpha = 0.5;
        for(int split = 1; split <= n_splits; split++) {
            System.out.println("Step : " + split);

            String data_folder_split = data_folder + "/" + split + "/";
            String monitoring_folder_split = monitoring_folder + "/" + split + "/";

            String results_folder_split = results_folder + "/" + split + "/";
            String drug_rbm_folder_split = drug_rbm_folder + "/" + split + "/";
            String target_rbm_folder_split = target_rbm_folder + "/" + split + "/";


            // We load drug_rbm data
            load_distinction_data_drug_rbm(data_folder_split);
            // We load target_rbm data
            load_distinction_data_target_rbm(data_folder_split);

            double [][][] formated_results = new double[numTargets][numClass][numDrugs];

            if(method == 1){
                DrugTargetCRBM target_rbm = DrugTargetCRBMTool.load(target_rbm_folder_split, numDrugs, hidUnits, lr, weightCost, momentum, numClass);
                DrugTargetCRBM drug_rbm = DrugTargetCRBMTool.load(drug_rbm_folder_split, numTargets, hidUnits, lr, weightCost, momentum, numClass);

                for(int p = 0; p < numTargets; p++){
                    double[][] input_target = { X_train_target_rbm_direct.getRow(p).toArray(),  X_train_target_rbm_indirect.getRow(p).toArray()};
                    boolean[] missing = X_train_target_rbm_isMissing.getRow(p).toBooleanArray();
                    double[][] probTarget = { probTarget_direct.getRow(p).toArray(),  probTarget_indirect.getRow(p).toArray()};

                    double[][] first_prediction = DrugTargetCRBMTool.predict(target_rbm, input_target, probTarget, missing);
                    for(int interaction = 0; interaction < numClass; interaction++){
                        for(int drug = 0; drug < numDrugs; drug++){
                            formated_results[p][interaction][drug] = first_prediction[interaction][drug];
                        }
                    }
                }

                for(int d = 0; d < numDrugs; d++){
                    double[][] input_drug = { X_train_drug_rbm_direct.getRow(d).toArray(),  X_train_drug_rbm_indirect.getRow(d).toArray()};
                    boolean[] missing = X_train_drug_rbm_isMissing.getRow(d).toBooleanArray();
                    double[][] probDrug = { probDrug_direct.getRow(d).toArray(),  probDrug_indirect.getRow(d).toArray()};

                    double[][] second_prediction = DrugTargetCRBMTool.predict(drug_rbm, input_drug, probDrug, missing);
                    for(int interaction = 0; interaction < numClass; interaction++){
                        for(int target = 0; target < numTargets; target++){
                            formated_results[target][interaction][d] = formated_results[target][interaction][d] * alpha
                                    + second_prediction[interaction][target] * (1 - alpha);
                        }
                    }
                }
            }else{
                DrugTargetCRBM target_rbm = DrugTargetCRBMTool.load(target_rbm_folder_split, numDrugs, hidUnits, lr, weightCost, momentum, 1);
                DrugTargetCRBM drug_rbm = DrugTargetCRBMTool.load(drug_rbm_folder_split, numTargets, hidUnits, lr, weightCost, momentum, 1);

                for(int p = 0; p < numTargets; p++){
                    boolean[] missing = X_train_target_rbm_isMissing.getRow(p).toBooleanArray();
                    double[][] input_target_direct = { X_train_target_rbm_direct.getRow(p).toArray() };
                    double[][] input_target_indirect = { X_train_target_rbm_indirect.getRow(p).toArray()};

                    double[][] probTarget_0 = { probTarget_direct.getRow(p).toArray() };
                    double[][] probTarget_1 = { probTarget_indirect.getRow(p).toArray()};

                    double[][] first_prediction_direct = DrugTargetCRBMTool.predict(target_rbm, input_target_direct, probTarget_0, missing);
                    double[][] first_prediction_indirect = DrugTargetCRBMTool.predict(target_rbm, input_target_indirect, probTarget_1, missing);

                    // Direct interaction prediction
                    for(int drug = 0; drug < numDrugs; drug++){
                        formated_results[p][0][drug] = first_prediction_direct[0][drug];
                    }
                    // Indirect interaction prediction
                    for(int drug = 0; drug < numDrugs; drug++){
                        formated_results[p][1][drug] = first_prediction_indirect[0][drug];
                    }
                }


                for(int d = 0; d < numDrugs; d++){
                    boolean[] missing = X_train_drug_rbm_isMissing.getRow(d).toBooleanArray();
                    double[][] input_drug_direct = { X_train_drug_rbm_direct.getRow(d).toArray() };
                    double[][] input_drug_indirect = { X_train_drug_rbm_indirect.getRow(d).toArray()};

                    double[][] probDrug_0 = { probDrug_direct.getRow(d).toArray() };
                    double[][] probDrug_1 = { probDrug_indirect.getRow(d).toArray()};

                    double[][] second_prediction_direct = DrugTargetCRBMTool.predict(drug_rbm, input_drug_direct, probDrug_0, missing);
                    double[][] second_prediction_indirect = DrugTargetCRBMTool.predict(drug_rbm, input_drug_indirect, probDrug_1, missing);

                    // Direct interaction prediction
                    for(int target = 0; target < numTargets; target++){
                        formated_results[target][0][d] = formated_results[target][0][d] * alpha
                                + second_prediction_direct[0][target] * (1 - alpha);
                    }

                    // Indirect interaction prediction
                    for(int target = 0; target < numTargets; target++){
                        formated_results[target][1][d] = formated_results[target][1][d] * alpha
                                + second_prediction_indirect[0][target] * (1 - alpha);
                    }
                }
            }

            File dir_results_folder = new File(results_folder_split);
            dir_results_folder.mkdirs();

            // We generate the evaluation file
            evaluate_distinction(numTargets, numDrugs, formated_results, results_folder_split);
        }

        System.exit(0);
    }

    private static void load_distinction_data_target_rbm(String data_folder){
        try {
            X_train_target_rbm_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_target_rbm.csv");
            X_train_target_rbm_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_target_rbm.csv");
            X_train_target_rbm_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_target_rbm.csv");

            probTarget_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_target_rbm.csv");
            probTarget_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_target_rbm.csv");

        } catch (IOException e) {
            System.out.println("Error while loading distinction the data: " + e.toString());
        }
    }

    private static void evaluate_distinction(int numTargets, int numDrugs, double [][][] formated_results, String results_folder){
        // ArrayList to contain all the true interactions (training and test data)
        ArrayList<String> y_true_all_direct = new ArrayList<String>();
        ArrayList<String> y_true_all_indirect = new ArrayList<String>();

        // ArrayList to contain the true probabilities for all the interactions (training and test data)
        ArrayList<String> prob_all_direct = new ArrayList<String>();
        ArrayList<String> prob_all_indirect = new ArrayList<String>();

        // ArrayList to contain the predictions of the interactions (training and test data)
        ArrayList<String> y_prediction_all_direct = new ArrayList<String>();
        ArrayList<String> y_prediction_all_indirect = new ArrayList<String>();

        // ArrayList to contain the true interactions (test data only)
        ArrayList<String> y_true_test_direct = new ArrayList<String>();
        ArrayList<String> y_true_test_indirect = new ArrayList<String>();

        // ArrayList to contain the probabilities of the interactions (test data only)
        ArrayList<String> prob_test_direct = new ArrayList<String>();
        ArrayList<String> prob_test_indirect = new ArrayList<String>();

        // ArrayList to contain the predictions of the interactions (test data only)
        ArrayList<String> y_prediction_test_direct = new ArrayList<String>();
        ArrayList<String> y_prediction_test_indirect = new ArrayList<String>();

        for(int p = 0; p < numTargets; p++){
            boolean[] missing = X_train_target_rbm_isMissing.getRow(p).toBooleanArray();
            double[][] input = { X_train_target_rbm_direct.getRow(p).toArray(),  X_train_target_rbm_indirect.getRow(p).toArray()};
            double[][] output = formated_results[p];

            for(int d = 0; d < numDrugs; d++){
                double pred_all_0 = 0.0;
                double pred_all_1 = 0.0;
                // Get training part
                y_true_all_direct.add(String.valueOf(input[0][d]));
                y_true_all_indirect.add(String.valueOf(input[1][d]));

                prob_all_direct.add(String.valueOf(output[0][d]));
                prob_all_indirect.add(String.valueOf(output[1][d]));

                if(output[0][d] > 0.5){
                    pred_all_0 = 1.0;
                }
                y_prediction_all_direct.add(String.valueOf(pred_all_0));

                if(output[1][d] > 0.5){
                    pred_all_1 = 1.0;
                }
                y_prediction_all_indirect.add(String.valueOf(pred_all_1));

                // Get testing part
                if ((missing[d] && (input[0][d] == 1.0 || input[1][d] == 1.0))){
                    double pred_test_0 = 0.0;
                    double pred_test_1 = 0.0;

                    y_true_test_direct.add(String.valueOf(input[0][d]));
                    y_true_test_indirect.add(String.valueOf(input[1][d]));

                    prob_test_direct.add(String.valueOf(output[0][d]));
                    prob_test_indirect.add(String.valueOf(output[1][d]));

                    if(output[0][d] > 0.5){
                        pred_test_0 = 1.0;
                    }
                    y_prediction_test_direct.add(String.valueOf(pred_test_0));

                    if(output[1][d] > 0.5){
                        pred_test_1 = 1.0;
                    }
                    y_prediction_test_indirect.add(String.valueOf(pred_test_1));
                }
            }
        }

        try {
            // We save y_true_all
            // 1. y_true_all_direct
            String y_true_all_direct_file = results_folder + "/y_true_all_direct.csv";
            FileWriter y_true_all_direct_writer = new FileWriter(y_true_all_direct_file);
            CSVUtils.writeLine(y_true_all_direct_writer, y_true_test_direct);
            y_true_all_direct_writer.flush();
            y_true_all_direct_writer.close();

            // 2. y_true_all_indirect
            String y_true_all_indirect_file = results_folder + "/y_true_all_indirect.csv";
            FileWriter y_true_all_indirect_writer = new FileWriter(y_true_all_indirect_file);
            CSVUtils.writeLine(y_true_all_indirect_writer, y_true_test_indirect);
            y_true_all_indirect_writer.flush();
            y_true_all_indirect_writer.close();

            // We save prob_all
            // 1. prob_all_direct
            String prob_all_direct_file = results_folder + "/prob_all_direct.csv";
            FileWriter prob_all_direct_writer = new FileWriter(prob_all_direct_file);
            CSVUtils.writeLine(prob_all_direct_writer, prob_all_direct);
            prob_all_direct_writer.flush();
            prob_all_direct_writer.close();

            // 2. prob_all_indirect
            String prob_all_indirect_file = results_folder + "/prob_all_indirect.csv";
            FileWriter prob_all_indirect_writer = new FileWriter(prob_all_indirect_file);
            CSVUtils.writeLine(prob_all_indirect_writer, prob_all_indirect);
            prob_all_indirect_writer.flush();
            prob_all_indirect_writer.close();

            // We save prediction_all
            // 1. y_prediction_all_direct
            String y_prediction_all_direct_file = results_folder + "/y_prediction_all_direct.csv";
            FileWriter y_prediction_all_direct_writer = new FileWriter(y_prediction_all_direct_file);
            CSVUtils.writeLine(y_prediction_all_direct_writer, y_prediction_all_direct);
            y_prediction_all_direct_writer.flush();
            y_prediction_all_direct_writer.close();

            // 2. y_prediction_all_indirect
            String y_prediction_all_indirect_file = results_folder + "/y_prediction_all_indirect.csv";
            FileWriter y_prediction_all_indirect_writer = new FileWriter(y_prediction_all_indirect_file);
            CSVUtils.writeLine(y_prediction_all_indirect_writer, y_prediction_all_indirect);
            y_prediction_all_indirect_writer.flush();
            y_prediction_all_indirect_writer.close();


            // We save y_true_test
            // 1. y_true_test_direct
            String y_true_test_direct_file = results_folder + "/y_true_test_direct.csv";
            FileWriter y_true_test_direct_writer = new FileWriter(y_true_test_direct_file);
            CSVUtils.writeLine(y_true_test_direct_writer, y_true_test_direct);
            y_true_test_direct_writer.flush();
            y_true_test_direct_writer.close();

            // 2. y_true_test_indirect
            String y_true_test_indirect_file = results_folder + "/y_true_test_indirect.csv";
            FileWriter y_true_test_indirect_writer = new FileWriter(y_true_test_indirect_file);
            CSVUtils.writeLine(y_true_test_indirect_writer, y_true_test_indirect);
            y_true_test_indirect_writer.flush();
            y_true_test_indirect_writer.close();

            // We save prob_test
            // 1. prob_test_direct
            String prob_test_direct_file = results_folder + "/prob_test_direct.csv";
            FileWriter prob_test_direct_writer = new FileWriter(prob_test_direct_file);
            CSVUtils.writeLine(prob_test_direct_writer, prob_test_direct);
            prob_test_direct_writer.flush();
            prob_test_direct_writer.close();

            // 2. prob_test_direct
            String prob_test_indirect_file = results_folder + "/prob_test_indirect.csv";
            FileWriter prob_test_indirect_writer = new FileWriter(prob_test_indirect_file);
            CSVUtils.writeLine(prob_test_indirect_writer, prob_test_indirect);
            prob_test_indirect_writer.flush();
            prob_test_indirect_writer.close();

            // We save prediction_test
            // 1. y_prediction_test_direct
            String y_prediction_test_direct_file = results_folder + "/y_prediction_test_direct.csv";
            FileWriter y_prediction_test_direct_writer = new FileWriter(y_prediction_test_direct_file);
            CSVUtils.writeLine(y_prediction_test_direct_writer, y_prediction_test_direct);
            y_prediction_test_direct_writer.flush();
            y_prediction_test_direct_writer.close();

            // 2. y_prediction_test_indirect
            String y_prediction_test_indirect_file = results_folder + "/y_prediction_test_indirect.csv";
            FileWriter y_prediction_test_indirect_writer = new FileWriter(y_prediction_test_indirect_file);
            CSVUtils.writeLine(y_prediction_test_indirect_writer, y_prediction_test_indirect);
            y_prediction_test_indirect_writer.flush();
            y_prediction_test_indirect_writer.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void load_distinction_data_drug_rbm(String data_folder){
        try {
            X_train_drug_rbm_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_drug_rbm.csv");
            X_train_drug_rbm_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_drug_rbm.csv");
            X_train_drug_rbm_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_drug_rbm.csv");

            probDrug_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_drug_rbm.csv");
            probDrug_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_drug_rbm.csv");

        } catch (IOException e) {
            System.out.println("Error while loading distinction the data: " + e.toString());
        }
    }
}
