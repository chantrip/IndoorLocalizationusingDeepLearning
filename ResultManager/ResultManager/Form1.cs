using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Oracle.ManagedDataAccess.Client;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using System.Windows.Forms;

namespace ResultManager
{
    public partial class Form1 : Form
    {
        public static Form1 currentInstant;
        public Form1()
        {
            InitializeComponent();
            currentInstant = this;
        }
        List<Job> historyJobList = new List<Job>();
        int jobRequestPointer = 1;
        public void setjobRequestPointer(int i)
        {
            this.Invoke(new Action(() =>
            {
                jobRequestPointer = i;
            }));
        }
        private void Form1_Load(object sender, EventArgs e)
        {

            startHttpServer();
            string[] hideColumns = { "RANDOM_SEED", "TRAINING_RATIO", "SAE_BIAS", "SAE_OPTIMIZER", "SAE_ACTIVATION",
                "SAE_LOSS", "CLASSIFIER_ACTIVATION", "CLASSIFIER_BIAS", "CLASSIFIER_OPTIMIZER",
                "CLASSIFIER_LOSS","ACC_BLD","ACC_FLR","ACC_BF","LOC_FAILURE" };
            foreach (DataGridViewColumn column in dataGridView1.Columns)
            {
                if (hideColumns.Contains(column.HeaderText))
                    column.Visible = false;
                else
                    column.Visible = true;
            }
            dataGridView1.AutoResizeColumns();

        }
        private bool isJobReadyToAssign(Job job)
        {
            bool isReady = true;
            foreach (Job j in historyJobList)
            {
                if (j.Equals(job) && j.status == Job.Status.ASSIGNED)
                {
                    isReady = false;
                    break;
                }
            }
            return isReady;
        }
        private void updateHistory(Job job)
        {
            foreach (Job j in historyJobList)
            {
                if (j.Equals(job))
                {
                    historyJobList.Remove(j);
                    break;
                }
            }
            historyJobList.Add(job);

            updateGridView();
        }
        public delegate void updateGridViewDelegate(string msg);
        public void updateGridView()
        {
            dataGridView1.Invoke(new Action(() =>
            {
                dataGridView1.Rows.Clear();
                foreach (Job j in historyJobList)
                {
                    List<String> columns = j.toValues();
                    columns.Add(j.status.ToString());
                    dataGridView1.Rows.Add(columns.ToArray());
                    switch (j.status)
                    {
                        case Job.Status.DONE:
                            dataGridView1.Rows[dataGridView1.Rows.GetLastRow(DataGridViewElementStates.None)]
                                .DefaultCellStyle.BackColor = Color.LightGreen;
                            break;
                        case Job.Status.ASSIGNED:
                            dataGridView1.Rows[dataGridView1.Rows.GetLastRow(DataGridViewElementStates.None)]
                                .DefaultCellStyle.BackColor = Color.LightSalmon;
                            break;
                        default:
                            dataGridView1.Rows[dataGridView1.Rows.GetLastRow(DataGridViewElementStates.None)]
                                .DefaultCellStyle.BackColor = Color.White;
                            break;
                    }

                }
                dataGridView1.AutoResizeColumns();
                int nRowIndex = dataGridView1.Rows.Count - 1;
                if (nRowIndex >= 0)
                {
                    dataGridView1.Rows[nRowIndex].Cells[0].Selected = true;
                    dataGridView1.FirstDisplayedScrollingRowIndex = nRowIndex;
                }
            }));
        }


        private async void startHttpServer()
        {
            if (!HttpListener.IsSupported)
            {
                MessageBox.Show(this, "HttpListener is not supported!");
            }

            HttpListener listener = new HttpListener();
            listener.Prefixes.Add("http://*/indoor/");
            listener.Start();

            do
            {
                // Note: The GetContext method blocks while waiting for a request. 
                HttpListenerContext context = await listener.GetContextAsync();
                HttpListenerRequest request = context.Request;
                HttpListenerResponse response = context.Response;


                StreamReader reader = new StreamReader(request.InputStream, request.ContentEncoding);
                String requestBody = reader.ReadToEnd();
                JObject JRequestBody = JObject.Parse(requestBody);
                string responseString = "";

                JObject jResponse = null;
                switch (JRequestBody["COMMAND"].Value<String>())
                {
                    case "REQUESTJOB":
                        Job job = null;
                        do
                        {
                            if (ResultDatabase.countPendingJobs() != 0)
                            {
                                job = ResultDatabase.getPendingJobAt(jobRequestPointer++);
                                if (job == null || !isJobReadyToAssign(job))
                                {
                                    jobRequestPointer = 1;
                                    jResponse = new JObject();
                                    jResponse.Add("RESPONSE", "WAIT");
                                    responseString = jResponse.ToString();
                                    break;
                                }
                                else
                                {
                                    job.status = Job.Status.ASSIGNED;
                                    job.trained_by = JRequestBody["NAME"].Value<String>();
                                    job.startTimer();
                                    updateHistory(job);

                                    jResponse = JObject.FromObject(job);
                                    jResponse.Add("RESPONSE", "ASSIGNED");
                                    responseString = jResponse.ToString();
                                    break;
                                }
                            }
                            else // countPendingJobs() = 0
                            {
                                jResponse = new JObject();
                                jResponse.Add("RESPONSE", "FINISHED");
                                responseString = jResponse.ToString();
                                break;
                            }
                        } while (true);
                        break;
                    case "SUBMITRESULT":
                        Job jobResult = null;
                        jResponse = new JObject();
                        try
                        {
                            jobResult = JRequestBody.ToObject<Job>();
                            ResultDatabase.updateResult(jobResult);
                            jobResult.status = Job.Status.DONE;
                            jResponse.Add("RESPONSE", "SUCCESS");
                        }
                        catch (Exception)
                        {
                            if (jobResult != null)
                                jobResult.status = Job.Status.WAITING;
                            jResponse.Add("RESPONSE", "FAIL");
                        }
                        if (jobResult != null)
                            updateHistory(jobResult);
                        responseString = jResponse.ToString();
                        break;
                    default:
                        jResponse = new JObject();
                        jResponse.Add("RESPONSE", "INVALIDCMD");
                        break;
                }

                responseString = jResponse.ToString();
                // Get a response stream and write the response to it.
                byte[] buffer = System.Text.Encoding.UTF8.GetBytes(responseString);
                response.ContentLength64 = buffer.Length;
                System.IO.Stream output = response.OutputStream;
                output.Write(buffer, 0, buffer.Length);

                output.Close();

            } while (true);
            //listener.Stop();
        }
    }
}
