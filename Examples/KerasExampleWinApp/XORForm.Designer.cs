namespace KerasExampleWinApp
{
    partial class XORForm
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
            this.btnTrain = new System.Windows.Forms.Button();
            this.txtTrainingResult = new System.Windows.Forms.TextBox();
            this.worker = new System.ComponentModel.BackgroundWorker();
            this.SuspendLayout();
            // 
            // btnTrain
            // 
            this.btnTrain.Location = new System.Drawing.Point(277, 74);
            this.btnTrain.Name = "btnTrain";
            this.btnTrain.Size = new System.Drawing.Size(143, 23);
            this.btnTrain.TabIndex = 0;
            this.btnTrain.Text = "Run Training";
            this.btnTrain.UseVisualStyleBackColor = true;
            this.btnTrain.Click += new System.EventHandler(this.btnTrain_Click);
            // 
            // txtTrainingResult
            // 
            this.txtTrainingResult.Location = new System.Drawing.Point(31, 116);
            this.txtTrainingResult.Multiline = true;
            this.txtTrainingResult.Name = "txtTrainingResult";
            this.txtTrainingResult.Size = new System.Drawing.Size(389, 642);
            this.txtTrainingResult.TabIndex = 1;
            // 
            // worker
            // 
            this.worker.WorkerReportsProgress = true;
            this.worker.DoWork += new System.ComponentModel.DoWorkEventHandler(this.woker_DoWork);
            // 
            // XORForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(862, 790);
            this.Controls.Add(this.txtTrainingResult);
            this.Controls.Add(this.btnTrain);
            this.Name = "XORForm";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnTrain;
        private System.Windows.Forms.TextBox txtTrainingResult;
        private System.ComponentModel.BackgroundWorker worker;
    }
}

