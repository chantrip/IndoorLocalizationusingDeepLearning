namespace ResultManager
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.RANDOM_SEED = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TRAINING_RATIO = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.EPOCHS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.BATCH_SIZE = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.N = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SCALING = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SAE_HIDDEN_LAYERS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SAE_ACTIVATION = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SAE_BIAS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SAE_OPTIMIZER = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SAE_LOSS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CLASSIFIER_HIDDEN_LAYERS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CLASSIFIER_ACTIVATION = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CLASSIFIER_BIAS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CLASSIFIER_OPTIMIZER = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.CLASSIFIER_LOSS = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.DROPOUT = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ACC_BLD = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ACC_FLR = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ACC_BF = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.LOC_FAILURE = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.MEAN_POS_ERR = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.MEAN_POS_ERR_WEIGHTED = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TRAINED_BY = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SUBMITTED_DATE = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TIME_SPENT = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Status = new System.Windows.Forms.DataGridViewTextBoxColumn();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            this.SuspendLayout();
            // 
            // dataGridView1
            // 
            this.dataGridView1.AllowUserToAddRows = false;
            this.dataGridView1.AllowUserToDeleteRows = false;
            this.dataGridView1.AllowUserToOrderColumns = true;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.RANDOM_SEED,
            this.TRAINING_RATIO,
            this.EPOCHS,
            this.BATCH_SIZE,
            this.N,
            this.SCALING,
            this.SAE_HIDDEN_LAYERS,
            this.SAE_ACTIVATION,
            this.SAE_BIAS,
            this.SAE_OPTIMIZER,
            this.SAE_LOSS,
            this.CLASSIFIER_HIDDEN_LAYERS,
            this.CLASSIFIER_ACTIVATION,
            this.CLASSIFIER_BIAS,
            this.CLASSIFIER_OPTIMIZER,
            this.CLASSIFIER_LOSS,
            this.DROPOUT,
            this.ACC_BLD,
            this.ACC_FLR,
            this.ACC_BF,
            this.LOC_FAILURE,
            this.MEAN_POS_ERR,
            this.MEAN_POS_ERR_WEIGHTED,
            this.TRAINED_BY,
            this.SUBMITTED_DATE,
            this.TIME_SPENT,
            this.Status});
            this.dataGridView1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.dataGridView1.Location = new System.Drawing.Point(0, 0);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.Size = new System.Drawing.Size(1314, 431);
            this.dataGridView1.TabIndex = 0;
            // 
            // RANDOM_SEED
            // 
            this.RANDOM_SEED.HeaderText = "RANDOM_SEED";
            this.RANDOM_SEED.Name = "RANDOM_SEED";
            this.RANDOM_SEED.ReadOnly = true;
            // 
            // TRAINING_RATIO
            // 
            this.TRAINING_RATIO.HeaderText = "TRAINING_RATIO";
            this.TRAINING_RATIO.Name = "TRAINING_RATIO";
            this.TRAINING_RATIO.ReadOnly = true;
            // 
            // EPOCHS
            // 
            this.EPOCHS.HeaderText = "EPOCHS";
            this.EPOCHS.Name = "EPOCHS";
            this.EPOCHS.ReadOnly = true;
            // 
            // BATCH_SIZE
            // 
            this.BATCH_SIZE.HeaderText = "BATCH_SIZE";
            this.BATCH_SIZE.Name = "BATCH_SIZE";
            this.BATCH_SIZE.ReadOnly = true;
            // 
            // N
            // 
            this.N.HeaderText = "N";
            this.N.Name = "N";
            this.N.ReadOnly = true;
            // 
            // SCALING
            // 
            this.SCALING.HeaderText = "SCALING";
            this.SCALING.Name = "SCALING";
            this.SCALING.ReadOnly = true;
            // 
            // SAE_HIDDEN_LAYERS
            // 
            this.SAE_HIDDEN_LAYERS.HeaderText = "SAE_HIDDEN_LAYERS";
            this.SAE_HIDDEN_LAYERS.Name = "SAE_HIDDEN_LAYERS";
            this.SAE_HIDDEN_LAYERS.ReadOnly = true;
            // 
            // SAE_ACTIVATION
            // 
            this.SAE_ACTIVATION.HeaderText = "SAE_ACTIVATION";
            this.SAE_ACTIVATION.Name = "SAE_ACTIVATION";
            this.SAE_ACTIVATION.ReadOnly = true;
            // 
            // SAE_BIAS
            // 
            this.SAE_BIAS.HeaderText = "SAE_BIAS";
            this.SAE_BIAS.Name = "SAE_BIAS";
            this.SAE_BIAS.ReadOnly = true;
            // 
            // SAE_OPTIMIZER
            // 
            this.SAE_OPTIMIZER.HeaderText = "SAE_OPTIMIZER";
            this.SAE_OPTIMIZER.Name = "SAE_OPTIMIZER";
            this.SAE_OPTIMIZER.ReadOnly = true;
            // 
            // SAE_LOSS
            // 
            this.SAE_LOSS.HeaderText = "SAE_LOSS";
            this.SAE_LOSS.Name = "SAE_LOSS";
            this.SAE_LOSS.ReadOnly = true;
            // 
            // CLASSIFIER_HIDDEN_LAYERS
            // 
            this.CLASSIFIER_HIDDEN_LAYERS.HeaderText = "CLASSIFIER_HIDDEN_LAYERS";
            this.CLASSIFIER_HIDDEN_LAYERS.Name = "CLASSIFIER_HIDDEN_LAYERS";
            this.CLASSIFIER_HIDDEN_LAYERS.ReadOnly = true;
            // 
            // CLASSIFIER_ACTIVATION
            // 
            this.CLASSIFIER_ACTIVATION.HeaderText = "CLASSIFIER_ACTIVATION";
            this.CLASSIFIER_ACTIVATION.Name = "CLASSIFIER_ACTIVATION";
            this.CLASSIFIER_ACTIVATION.ReadOnly = true;
            // 
            // CLASSIFIER_BIAS
            // 
            this.CLASSIFIER_BIAS.HeaderText = "CLASSIFIER_BIAS";
            this.CLASSIFIER_BIAS.Name = "CLASSIFIER_BIAS";
            this.CLASSIFIER_BIAS.ReadOnly = true;
            // 
            // CLASSIFIER_OPTIMIZER
            // 
            this.CLASSIFIER_OPTIMIZER.HeaderText = "CLASSIFIER_OPTIMIZER";
            this.CLASSIFIER_OPTIMIZER.Name = "CLASSIFIER_OPTIMIZER";
            this.CLASSIFIER_OPTIMIZER.ReadOnly = true;
            // 
            // CLASSIFIER_LOSS
            // 
            this.CLASSIFIER_LOSS.HeaderText = "CLASSIFIER_LOSS";
            this.CLASSIFIER_LOSS.Name = "CLASSIFIER_LOSS";
            this.CLASSIFIER_LOSS.ReadOnly = true;
            // 
            // DROPOUT
            // 
            this.DROPOUT.HeaderText = "DROPOUT";
            this.DROPOUT.Name = "DROPOUT";
            // 
            // ACC_BLD
            // 
            this.ACC_BLD.HeaderText = "ACC_BLD";
            this.ACC_BLD.Name = "ACC_BLD";
            this.ACC_BLD.ReadOnly = true;
            // 
            // ACC_FLR
            // 
            this.ACC_FLR.HeaderText = "ACC_FLR";
            this.ACC_FLR.Name = "ACC_FLR";
            this.ACC_FLR.ReadOnly = true;
            // 
            // ACC_BF
            // 
            this.ACC_BF.HeaderText = "ACC_BF";
            this.ACC_BF.Name = "ACC_BF";
            this.ACC_BF.ReadOnly = true;
            // 
            // LOC_FAILURE
            // 
            this.LOC_FAILURE.HeaderText = "LOC_FAILURE";
            this.LOC_FAILURE.Name = "LOC_FAILURE";
            this.LOC_FAILURE.ReadOnly = true;
            // 
            // MEAN_POS_ERR
            // 
            this.MEAN_POS_ERR.HeaderText = "MEAN_POS_ERR";
            this.MEAN_POS_ERR.Name = "MEAN_POS_ERR";
            this.MEAN_POS_ERR.ReadOnly = true;
            // 
            // MEAN_POS_ERR_WEIGHTED
            // 
            this.MEAN_POS_ERR_WEIGHTED.HeaderText = "MEAN_POS_ERR_WEIGHTED";
            this.MEAN_POS_ERR_WEIGHTED.Name = "MEAN_POS_ERR_WEIGHTED";
            this.MEAN_POS_ERR_WEIGHTED.ReadOnly = true;
            // 
            // TRAINED_BY
            // 
            this.TRAINED_BY.HeaderText = "TRAINED_BY";
            this.TRAINED_BY.Name = "TRAINED_BY";
            this.TRAINED_BY.ReadOnly = true;
            // 
            // SUBMITTED_DATE
            // 
            this.SUBMITTED_DATE.HeaderText = "SUBMITTED_DATE";
            this.SUBMITTED_DATE.Name = "SUBMITTED_DATE";
            this.SUBMITTED_DATE.ReadOnly = true;
            // 
            // TIME_SPENT
            // 
            this.TIME_SPENT.HeaderText = "TIME_SPENT(s)";
            this.TIME_SPENT.Name = "TIME_SPENT";
            this.TIME_SPENT.ReadOnly = true;
            // 
            // Status
            // 
            this.Status.HeaderText = "Status";
            this.Status.Name = "Status";
            this.Status.ReadOnly = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1314, 431);
            this.Controls.Add(this.dataGridView1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.DataGridViewTextBoxColumn RANDOM_SEED;
        private System.Windows.Forms.DataGridViewTextBoxColumn TRAINING_RATIO;
        private System.Windows.Forms.DataGridViewTextBoxColumn EPOCHS;
        private System.Windows.Forms.DataGridViewTextBoxColumn BATCH_SIZE;
        private System.Windows.Forms.DataGridViewTextBoxColumn N;
        private System.Windows.Forms.DataGridViewTextBoxColumn SCALING;
        private System.Windows.Forms.DataGridViewTextBoxColumn SAE_HIDDEN_LAYERS;
        private System.Windows.Forms.DataGridViewTextBoxColumn SAE_ACTIVATION;
        private System.Windows.Forms.DataGridViewTextBoxColumn SAE_BIAS;
        private System.Windows.Forms.DataGridViewTextBoxColumn SAE_OPTIMIZER;
        private System.Windows.Forms.DataGridViewTextBoxColumn SAE_LOSS;
        private System.Windows.Forms.DataGridViewTextBoxColumn CLASSIFIER_HIDDEN_LAYERS;
        private System.Windows.Forms.DataGridViewTextBoxColumn CLASSIFIER_ACTIVATION;
        private System.Windows.Forms.DataGridViewTextBoxColumn CLASSIFIER_BIAS;
        private System.Windows.Forms.DataGridViewTextBoxColumn CLASSIFIER_OPTIMIZER;
        private System.Windows.Forms.DataGridViewTextBoxColumn CLASSIFIER_LOSS;
        private System.Windows.Forms.DataGridViewTextBoxColumn DROPOUT;
        private System.Windows.Forms.DataGridViewTextBoxColumn ACC_BLD;
        private System.Windows.Forms.DataGridViewTextBoxColumn ACC_FLR;
        private System.Windows.Forms.DataGridViewTextBoxColumn ACC_BF;
        private System.Windows.Forms.DataGridViewTextBoxColumn LOC_FAILURE;
        private System.Windows.Forms.DataGridViewTextBoxColumn MEAN_POS_ERR;
        private System.Windows.Forms.DataGridViewTextBoxColumn MEAN_POS_ERR_WEIGHTED;
        private System.Windows.Forms.DataGridViewTextBoxColumn TRAINED_BY;
        private System.Windows.Forms.DataGridViewTextBoxColumn SUBMITTED_DATE;
        private System.Windows.Forms.DataGridViewTextBoxColumn TIME_SPENT;
        private System.Windows.Forms.DataGridViewTextBoxColumn Status;
    }
}

