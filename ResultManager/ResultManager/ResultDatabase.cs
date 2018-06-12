using Oracle.ManagedDataAccess.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ResultManager
{
    class ResultDatabase
    {
        private static OracleConnection connect()
        {
            //string constr = "user id=tete;password=3138;data source=novasoft-th.com/orcl";
            string constr = "user id=tete;password=3138;data source=localhost/orcl";
            OracleConnection con = new OracleConnection(constr);
            con.Open();
            return con;
        }
        private static List<Job> convertOracleReaderToJobList(OracleDataReader dr, Boolean isResult = true)
        {
            List<Job> retList = new List<Job>();
            while (dr.Read())
            {
                Job job = new Job();
                job.random_seed = dr.GetInt32(dr.GetOrdinal("random_seed"));
                job.training_ratio = dr.GetDouble(dr.GetOrdinal("training_ratio"));
                job.epochs = dr.GetInt32(dr.GetOrdinal("epochs"));
                job.batch_size = dr.GetInt32(dr.GetOrdinal("batch_size"));
                job.N = dr.GetInt32(dr.GetOrdinal("N"));
                job.scaling = dr.GetDouble(dr.GetOrdinal("scaling"));
                job.sae_hidden_layers = dr.GetString(dr.GetOrdinal("sae_hidden_layers"));
                job.sae_activation = dr.GetString(dr.GetOrdinal("sae_activation"));
                job.sae_bias = dr.GetString(dr.GetOrdinal("sae_bias"));
                job.sae_optimizer = dr.GetString(dr.GetOrdinal("sae_optimizer"));
                job.sae_loss = dr.GetString(dr.GetOrdinal("sae_loss"));
                job.classifier_hidden_layers = dr.GetString(dr.GetOrdinal("classifier_hidden_layers"));
                job.classifier_activation = dr.GetString(dr.GetOrdinal("classifier_activation"));
                job.classifier_bias = dr.GetString(dr.GetOrdinal("classifier_bias"));
                job.classifier_optimizer = dr.GetString(dr.GetOrdinal("classifier_optimizer"));
                job.classifier_loss = dr.GetString(dr.GetOrdinal("classifier_loss"));
                job.dropout = dr.GetDouble(dr.GetOrdinal("dropout"));
                if (isResult)
                {
                    job.acc_bld = dr.GetDouble(dr.GetOrdinal("acc_bld"));
                    job.acc_flr = dr.GetDouble(dr.GetOrdinal("acc_flr"));
                    job.acc_bf = dr.GetDouble(dr.GetOrdinal("acc_bf"));
                    job.loc_failure = dr.GetDouble(dr.GetOrdinal("loc_failure"));
                    job.mean_pos_err = dr.GetDouble(dr.GetOrdinal("mean_pos_err"));
                    job.mean_pos_err_weighted = dr.GetDouble(dr.GetOrdinal("mean_pos_err_weighted"));
                    job.trained_by = dr.GetString(dr.GetOrdinal("trained_by"));
                    job.submitted_date = dr.GetDateTime(dr.GetOrdinal("submitted_date"));
                    job.time_spent = dr.GetInt32(dr.GetOrdinal("time_spent"));
                }
                retList.Add(job);
            }
            return retList;
        }

        public static List<Job> getAllPendingJobs()
        {
            OracleConnection conn = connect();
            List<Job> retList = new List<Job>();
            OracleCommand cmd = new OracleCommand("SELECT * FROM indoorloc WHERE MEAN_POS_ERR_WEIGHTED IS NULL", conn);
            conn.Close();
            return convertOracleReaderToJobList(cmd.ExecuteReader(), false);
        }
        public static Job getPendingJobAt(int i)
        {
            OracleConnection conn = connect();
            List<Job> retList = new List<Job>();
            String sql = String.Format("SELECT * FROM ( " +
                " SELECT " +
                "ROW_NUMBER() OVER (ORDER BY CLASSIFIER_HIDDEN_LAYERS DESC) AS rownumber, " +
                "RANDOM_SEED,TRAINING_RATIO,EPOCHS,BATCH_SIZE,N,SCALING,SAE_HIDDEN_LAYERS,SAE_ACTIVATION,SAE_BIAS,SAE_OPTIMIZER,SAE_LOSS,CLASSIFIER_HIDDEN_LAYERS,CLASSIFIER_ACTIVATION,CLASSIFIER_BIAS, " +
                "CLASSIFIER_OPTIMIZER,CLASSIFIER_LOSS,DROPOUT,ACC_BLD,ACC_FLR,ACC_BF,LOC_FAILURE,MEAN_POS_ERR,MEAN_POS_ERR_WEIGHTED,TRAINED_BY,SUBMITTED_DATE,TIME_SPENT " +
                "FROM INDOORLOC WHERE MEAN_POS_ERR_WEIGHTED IS NULL) " +
                "WHERE rownumber = {0}", i.ToString());
            OracleCommand cmd = new OracleCommand(sql, conn);
            OracleDataReader reader = cmd.ExecuteReader();
            try
            {
                if (reader.HasRows == false)
                    return null;
                else
                    return convertOracleReaderToJobList(reader, false).First(); ;
            }
            catch (Exception)
            {

                throw;
            }
            finally
            {
                conn.Close();
            }


        }
        public static int countPendingJobs()
        {
            OracleConnection conn = connect();
            OracleCommand cmd = new OracleCommand("SELECT COUNT(*) AS \"COUNT\" FROM indoorloc   WHERE MEAN_POS_ERR_WEIGHTED IS NULL", conn);
            OracleDataReader reader = cmd.ExecuteReader();
            reader.Read();
            int rowCount = reader.GetInt32(reader.GetOrdinal("COUNT"));
            conn.Close();
            return rowCount;
        }
        public static void updateResult(Job job)
        {
            OracleConnection conn = connect();
            OracleCommand cmd = new OracleCommand("UPDATE indoorloc SET " +
                "acc_bld = :acc_bld," +
                "acc_flr = :acc_flr," +
                "acc_bf = :acc_bf," +
                "loc_failure = :loc_failure," +
                "mean_pos_err = :mean_pos_err," +
                "mean_pos_err_weighted = :mean_pos_err_weighted," +
                "trained_by = :trained_by," +
                "submitted_date = :submitted_date," +
                "time_spent = :time_spent" +

                " WHERE " +
                "random_seed = :random_seed AND " +
                "training_ratio = :training_ratio AND " +
                "epochs = :epochs AND " +
                "batch_size = :batch_size AND " +
                "N = :N AND " +
                "scaling = :scaling AND " +
                "sae_hidden_layers = :sae_hidden_layers AND " +
                "sae_activation = :sae_activation AND " +
                "sae_bias = :sae_bias AND " +
                "sae_optimizer = :sae_optimizer AND " +
                "sae_loss = :sae_loss AND " +
                "classifier_hidden_layers = :classifier_hidden_layers AND " +
                "classifier_activation = :classifier_activation AND " +
                "classifier_bias = :classifier_bias AND " +
                "classifier_optimizer = :classifier_optimizer AND " +
                "classifier_loss = :classifier_loss AND " +
                "dropout = :dropout", conn);

            cmd.Parameters.Add("acc_bld", job.acc_bld);
            cmd.Parameters.Add("acc_flr", job.acc_flr);
            cmd.Parameters.Add("acc_bf", job.acc_bf);
            cmd.Parameters.Add("loc_failure", job.loc_failure);
            cmd.Parameters.Add("mean_pos_err", job.mean_pos_err);
            cmd.Parameters.Add("mean_pos_err_weighted", job.mean_pos_err_weighted);
            cmd.Parameters.Add("trained_by", job.trained_by);
            cmd.Parameters.Add("submitted_date", job.submitted_date);
            cmd.Parameters.Add("time_spent", job.time_spent);

            cmd.Parameters.Add("random_seed", job.random_seed);
            cmd.Parameters.Add("training_ratio", job.training_ratio);
            cmd.Parameters.Add("epochs", job.epochs);
            cmd.Parameters.Add("batch_size", job.batch_size);
            cmd.Parameters.Add("N", job.N);
            cmd.Parameters.Add("scaling", job.scaling);
            cmd.Parameters.Add("sae_hidden_layers", job.sae_hidden_layers);
            cmd.Parameters.Add("sae_activation", job.sae_activation);
            cmd.Parameters.Add("sae_bias", job.sae_bias.ToUpper());
            cmd.Parameters.Add("sae_optimizer", job.sae_optimizer);
            cmd.Parameters.Add("sae_loss", job.sae_loss);
            cmd.Parameters.Add("classifier_hidden_layers", job.classifier_hidden_layers);
            cmd.Parameters.Add("classifier_activation", job.classifier_activation);
            cmd.Parameters.Add("classifier_bias", job.classifier_bias.ToUpper());
            cmd.Parameters.Add("classifier_optimizer", job.classifier_optimizer);
            cmd.Parameters.Add("classifier_loss", job.classifier_loss);
            cmd.Parameters.Add("dropout", job.dropout);
            cmd.ExecuteNonQuery();
            conn.Close();
        }


    }
}
