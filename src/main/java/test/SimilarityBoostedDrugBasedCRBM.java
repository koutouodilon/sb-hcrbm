package test;

import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;
import sbhcrbm.DrugTargetCRBM;
import sbhcrbm.DrugTargetCRBMTool;
import utils.CSVUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Similarity Boosted Drug-based CRBM
 */
public class SimilarityBoostedDrugBasedCRBM {
    public static double[][][] X_train;

    public static DoubleMatrix X_train_direct;
    public static DoubleMatrix X_train_indirect;
    public static DoubleMatrix X_train_isMissing;

    public static double[][][] probDrug;
    public static DoubleMatrix probDrug_direct;
    public static DoubleMatrix probDrug_indirect;

    public static boolean[][] isMissing;

    public static double[][][] X_test;

    public static DoubleMatrix X_test_direct;
    public static DoubleMatrix X_test_indirect;
    public static DoubleMatrix X_test_isMissing;

    public static double[][][] probDrugTesting;
    public static DoubleMatrix probDrugTesting_direct;
    public static DoubleMatrix probDrugTesting_indirect;

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
        int numDrugs = Integer.parseInt(args[13]);
        int numTargets = Integer.parseInt(args[14]);
        int numClass = Integer.parseInt(args[15]);

        char backup_option = args[16].charAt(0);

        if(backup_option == 'y'){
            File f = new File("./backup/" + backup_folder);
            if (!f.exists()) {
                System.out.println("No Backup folder found! Will start training the model.");
                backup_option = 'n';
            }
            try {
                FileUtils.deleteDirectory(new File(results_folder));
                File dir_results_folder = new File(results_folder);
                dir_results_folder.mkdirs();
            } catch (IOException e) {
                System.out.println("Error while deleting the results folder: " + e.toString() + "! Existing.");
                System.exit(-2);
            }
        }

        for(int split = 1; split <= n_splits; split++) {
            System.out.println("Step : " + split);

            String data_folder_split = data_folder + "/" + split + "/";
            String monitoring_folder_split = monitoring_folder + "/" + split + "/";
            String backup_folder_split = backup_folder + "/" + split + "/";
            String results_folder_split = results_folder + "/" + split + "/";

            // We load the data
            if(method == 1){
                load_distinction_data(data_folder_split);
            }else{
                load_no_distinction_data(data_folder_split);
            }

            DrugTargetCRBM d_crbm;

            if(backup_option == 'y'){
                d_crbm = DrugTargetCRBMTool.load("./backup/" + backup_folder_split, numTargets, hidUnits, lr, weightCost, momentum, numClass);
                File dir = new File(results_folder_split);
                dir.mkdirs();
            }else{
                // We train the model
                d_crbm = DrugTargetCRBMTool.fit(X_train, probDrug, isMissing, hidUnits, epochs, lr, momentum, weightCost, monitoring_folder_split);

                // We save the trained model
                d_crbm.save(backup_folder_split);
            }

            // We generate the evaluation file
            if(method == 1){
                evaluate_distinction(d_crbm, results_folder_split);
            }else{
                evaluate_no_distinction(d_crbm, results_folder_split);
            }
        }

        System.exit(0);
    }

    /**
     * Method to load the data with distinction
     *
     * @param data_folder
     */
    private static void load_distinction_data(String data_folder){
        try {
            X_train_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_drug_rbm.csv");
            X_train_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_drug_rbm.csv");
            X_train_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_drug_rbm_training.csv");

            probDrug_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_probDrugTraining.csv");
            probDrug_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_probDrugTraining.csv");


            X_train = new double[2][][];
            X_train[0] = X_train_direct.toArray2();
            X_train[1] = X_train_indirect.toArray2();

            isMissing = X_train_isMissing.toBooleanArray2();

            probDrug = new double[2][][];
            probDrug[0] = probDrug_direct.toArray2();
            probDrug[1] = probDrug_indirect.toArray2();


            X_test_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_test_drug_rbm.csv");
            X_test_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_test_drug_rbm.csv");
            X_test_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_drug_rbm_testing.csv");

            probDrugTesting_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_probDrugTesting.csv");
            probDrugTesting_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_probDrugTesting.csv");

            X_test = new double[2][][];
            X_test[0] = X_test_direct.toArray2();
            X_test[1] = X_test_indirect.toArray2();

            probDrugTesting = new double[2][][];
            probDrugTesting[0] = probDrugTesting_direct.toArray2();
            probDrugTesting[1] = probDrugTesting_indirect.toArray2();

        } catch (IOException e) {
            System.out.println("Error while loading distinction the data: " + e.toString());
        }
    }

    /**
     *
     * Method to load the data without distinction
     *
     * @param data_folder
     */
    private static void load_no_distinction_data(String data_folder) {
        try {

            X_train_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_train_drug_rbm.csv");
            X_train_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_train_drug_rbm.csv");
            X_train_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_drug_rbm_training.csv");

            probDrug_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_probDrugTraining.csv");
            probDrug_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_probDrugTraining.csv");

            X_train = new double[1][][];
            X_train[0] = X_train_direct.add(X_train_indirect).toArray2();

            isMissing = X_train_isMissing.toBooleanArray2();

            probDrug = new double[1][][];
            probDrug[0] = probDrug_direct.add(probDrug_indirect).toArray2();

            X_test_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_X_test_drug_rbm.csv");
            X_test_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_X_test_drug_rbm.csv");
            X_test_isMissing = DoubleMatrix.loadCSVFile(data_folder + "df_isMissing_drug_rbm_testing.csv");

            probDrugTesting_direct = DoubleMatrix.loadCSVFile(data_folder + "0_df_probDrugTesting.csv");
            probDrugTesting_indirect = DoubleMatrix.loadCSVFile(data_folder + "1_df_probDrugTesting.csv");

            X_test = new double[2][][];
            X_test[0] = X_test_direct.toArray2();
            X_test[1] = X_test_indirect.toArray2();

            probDrugTesting = new double[2][][];
            probDrugTesting[0] = probDrugTesting_direct.toArray2();
            probDrugTesting[1] = probDrugTesting_indirect.toArray2();

        } catch (IOException e) {
            System.out.println("Error while loading no distinction the data: " + e.toString());
        }
    }

    /**
     * Method to evaluate the D-CRBM with distinction
     * @param crbm
     * @param results_folder
     */
    private static void evaluate_distinction(DrugTargetCRBM crbm, String results_folder){
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

        for(int k = 0; k < X_test_direct.getRows(); k++){
            boolean[] missing = X_test_isMissing.getRow(k).toBooleanArray();

            double[][] input = { X_test_direct.getRow(k).toArray(), X_test_indirect.getRow(k).toArray() };
            double[][] prob = { probDrugTesting_direct.getRow(k).toArray(), probDrugTesting_indirect.getRow(k).toArray() };
            double[][] output = DrugTargetCRBMTool.predict(crbm, input, prob, missing);

            if(output == null) {
                System.out.println("Input format is not right");
                System.exit(-1);
            }
            for(int col = 0; col < output[0].length; col++) {
                double pred_all_0 = 0.0;
                double pred_all_1 = 0.0;
                // Get training part
                y_true_all_direct.add(String.valueOf(input[0][col]));
                y_true_all_indirect.add(String.valueOf(input[1][col]));

                prob_all_direct.add(String.valueOf(output[0][col]));
                prob_all_indirect.add(String.valueOf(output[1][col]));

                if(output[0][col] > 0.5){
                    pred_all_0 = 1.0;
                }
                y_prediction_all_direct.add(String.valueOf(pred_all_0));

                if(output[1][col] > 0.5){
                    pred_all_1 = 1.0;
                }
                y_prediction_all_indirect.add(String.valueOf(pred_all_1));

                // Get testing part
                if ((missing[col] && (input[0][col] == 1.0 || input[1][col] == 1.0))){
                    double pred_test_0 = 0.0;
                    double pred_test_1 = 0.0;

                    y_true_test_direct.add(String.valueOf(input[0][col]));
                    y_true_test_indirect.add(String.valueOf(input[1][col]));

                    prob_test_direct.add(String.valueOf(output[0][col]));
                    prob_test_indirect.add(String.valueOf(output[1][col]));

                    if(output[0][col] > 0.5){
                        pred_test_0 = 1.0;
                    }
                    y_prediction_test_direct.add(String.valueOf(pred_test_0));

                    if(output[1][col] > 0.5){
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
            CSVUtils.writeLine(y_true_all_direct_writer, y_true_all_direct);
            y_true_all_direct_writer.flush();
            y_true_all_direct_writer.close();

            // 2. y_true_all_indirect
            String y_true_all_indirect_file = results_folder + "/y_true_all_indirect.csv";
            FileWriter y_true_all_indirect_writer = new FileWriter(y_true_all_indirect_file);
            CSVUtils.writeLine(y_true_all_indirect_writer, y_true_all_indirect);
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


    /**
     * Method to evaluate the D-CRBM without distinction
     * @param crbm
     * @param results_folder
     */
    private static void evaluate_no_distinction(DrugTargetCRBM crbm, String results_folder){
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

        for(int k = 0; k < X_test_direct.getRows(); k++){
            boolean[] missing = X_test_isMissing.getRow(k).toBooleanArray();

            double[][] input_direct = { X_test_direct.getRow(k).toArray() };
            double[][] prob_direct = { probDrugTesting_direct.getRow(k).toArray() };
            double[][] output_direct = DrugTargetCRBMTool.predict(crbm, input_direct, prob_direct, missing);

            double[][] input_indirect = { X_test_indirect.getRow(k).toArray() };
            double[][] prob_indirect = { probDrugTesting_indirect.getRow(k).toArray() };
            double[][] output_indirect = DrugTargetCRBMTool.predict(crbm, input_indirect, prob_indirect, missing);

            if(output_direct == null || output_indirect == null) {
                System.out.println("Input format is not right");
                System.exit(-1);
            }

            for(int col = 0; col < output_direct[0].length; col++) {
                double pred_all_0 = 0.0;
                double pred_all_1 = 0.0;
                // Get training part
                y_true_all_direct.add(String.valueOf(input_direct[0][col]));
                y_true_all_indirect.add(String.valueOf(input_indirect[0][col]));

                prob_all_direct.add(String.valueOf(output_direct[0][col]));
                prob_all_indirect.add(String.valueOf(output_indirect[0][col]));

                if(output_direct[0][col] > 0.5){
                    pred_all_0 = 1.0;
                }
                y_prediction_all_direct.add(String.valueOf(pred_all_0));

                if(output_indirect[0][col] > 0.5){
                    pred_all_1 = 1.0;
                }
                y_prediction_all_indirect.add(String.valueOf(pred_all_1));

                // Get testing part
                if ((missing[col] && (input_direct[0][col] == 1.0 || input_indirect[0][col] == 1.0))){
                    double pred_test_0 = 0.0;
                    double pred_test_1 = 0.0;

                    y_true_test_direct.add(String.valueOf(input_direct[0][col]));
                    y_true_test_indirect.add(String.valueOf(input_indirect[0][col]));

                    prob_test_direct.add(String.valueOf(output_direct[0][col]));
                    prob_test_indirect.add(String.valueOf(output_indirect[0][col]));

                    if(output_direct[0][col] > 0.5){
                        pred_test_0 = 1.0;
                    }
                    y_prediction_test_direct.add(String.valueOf(pred_test_0));

                    if(output_indirect[0][col] > 0.5){
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
            CSVUtils.writeLine(y_true_all_direct_writer, y_true_all_direct);
            y_true_all_direct_writer.flush();
            y_true_all_direct_writer.close();

            // 2. y_true_all_indirect
            String y_true_all_indirect_file = results_folder + "/y_true_all_indirect.csv";
            FileWriter y_true_all_indirect_writer = new FileWriter(y_true_all_indirect_file);
            CSVUtils.writeLine(y_true_all_indirect_writer, y_true_all_indirect);
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

}
