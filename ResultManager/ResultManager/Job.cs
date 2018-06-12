using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace ResultManager
{
    public class Job
    {
        public int random_seed;
        public double training_ratio;
        public int epochs;
        public int batch_size;
        public int N;
        public double scaling;
        public string sae_hidden_layers;
        public string sae_activation;
        public string sae_bias;
        public string sae_optimizer;
        public string sae_loss;
        public string classifier_hidden_layers;
        public string classifier_activation;
        public string classifier_bias;
        public string classifier_optimizer;
        public string classifier_loss;
        public double dropout;

        public double acc_bld;
        public double acc_flr;
        public double acc_bf;
        public double loc_failure;
        public double mean_pos_err;
        public double mean_pos_err_weighted;
        public string trained_by;
        public int time_spent;
        public DateTime submitted_date;

        public enum Status
        {
            WAITING,
            ASSIGNED,
            DONE
        }
        [NonSerialized]
        public Timer timer;
        [NonSerialized]
        public Status status;
        [NonSerialized]
        const int MAXIMUM_TIME = 50 * 60 * 1000; //20 minutes

        void New()
        {
            status = Status.WAITING;

        }
        private void timeEscaped(Object source, System.Timers.ElapsedEventArgs e)
        {
            status = Status.WAITING;
            timer.Stop();
            timer.Dispose();

            Form1.currentInstant.updateGridView();
            Form1.currentInstant.setjobRequestPointer(1);

        }
        public void startTimer()
        {
            timer = new Timer(MAXIMUM_TIME);
            timer.Elapsed += timeEscaped;
            timer.Enabled = true;
            timer.Start();
        }
        public override bool Equals(object _obj)
        {
            if (_obj is Job)
            {
                Job obj = (Job)_obj;
                bool ret = obj.random_seed == random_seed && obj.training_ratio == training_ratio && obj.epochs == epochs && obj.batch_size == batch_size &&
                   obj.N == N && obj.scaling == scaling && obj.sae_hidden_layers == sae_hidden_layers && obj.sae_activation == sae_activation && obj.sae_bias.ToUpper() == sae_bias.ToUpper() &&
                   obj.sae_optimizer == sae_optimizer && obj.sae_loss == sae_loss && obj.classifier_hidden_layers == classifier_hidden_layers && obj.classifier_activation == classifier_activation &&
                   obj.classifier_bias.ToUpper() == classifier_bias.ToUpper() && obj.classifier_optimizer == classifier_optimizer && obj.classifier_loss == classifier_loss && obj.dropout == dropout;
                return ret;
            }
            else
            {
                return false;
            }



        }

        public List<String> toValues()
        {
            String[] array = new String[] { random_seed.ToString (), training_ratio.ToString (), epochs.ToString (), batch_size.ToString (),N.ToString (),
                scaling.ToString (),sae_hidden_layers,sae_activation,sae_bias,sae_optimizer,sae_loss,classifier_hidden_layers,classifier_activation
                ,classifier_bias,classifier_optimizer,classifier_loss,dropout.ToString (),acc_bld.ToString (),acc_flr.ToString ()
                ,acc_bf.ToString (),loc_failure.ToString (),mean_pos_err.ToString (),mean_pos_err_weighted.ToString (),
                trained_by,submitted_date.ToString (),time_spent.ToString () };
            return new List<String>(array);
        }
    }
}
