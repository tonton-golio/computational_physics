import streamlit as st
from utils.utils_global import *

from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
def landing_page():
    ''
    """# Applied Machine Learning"""


def lecture_2():
    ''
    "# Lecture 2: April 26, 2023"
    
    def loss_functions_():
        "## Loss Functions"
        # classification
        st.markdown(r"""### Classification""")
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown(r"""
            We have a couple different loss functions show on the right
            For **classification** we have
            * Zero-One Loss
                * zero if wrong classification, one if correct
            * Hinge Loss
                $$
                    \mathcal{L} = \max(0, 1-y_n \hat{y}_n)
                $$
            * Binary Cross Entropy (logistic loss or log loss)
                $$
                    \mathcal{L} = -\frac{1}{N}\sum_{n=1}^N \left(y_n \log \hat{y}_n + (1-y_n) \log (1-\hat{y}_n)\right)
                $$

            #### Unbalanced Data

            If we are dealing with unbalanced data (e.g. 99% of the data is one class, 1% is the other), we should be careful in picking a loss function. Approprate loss functions include
            * F1 Score
            * Binary Cross Entropy
            """)
        
        x = np.linspace(-3,3,100)
        zero_one = np.where(x<0, 1, 0)
        hinge = np.where(x<1, 1-x, 0)
        exponential = np.exp(-x)*.1
        binary_cross_entropy = -np.log(1/(1+np.exp(-x)))
        fig = plt.figure(figsize=(6,4))
        plt.plot(x, zero_one, label='Zero-One')
        plt.plot(x, hinge, label='Hinge')
        plt.plot(x, exponential, label='Exponential')
        plt.plot(x, binary_cross_entropy, label='Binary Cross Entropy')
        
        plt.legend()
        plt.title('Loss Functions')
        plt.xlabel('x')
        plt.ylabel('Loss')
        plt.grid()
        with cols[1]:
            st.pyplot(fig)

        st.markdown(r"""
        ### Regression
        Use one of the following
        * Mean Squared Error
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N (y_n - \hat{y}_n)^2
            $$
        * Mean Absolute Error -> problematic because not differentiable at 0
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N |y_n - \hat{y}_n|
            $$
        * Huber Loss -> a combination of the two above, differentiable at 0
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \begin{cases}
                    \frac{1}{2}(y_n - \hat{y}_n)^2 & \text{if } |y_n - \hat{y}_n| \leq \delta \\
                    \delta |y_n - \hat{y}_n| - \frac{1}{2}\delta^2 & \text{otherwise}
                \end{cases}
            $$
        * Log-Cosh Loss -> industry standard, double differentiable everywhere
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \log(\cosh(y_n - \hat{y}_n))
            $$
        * Quantile Loss
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \begin{cases}
                    \alpha |y_n - \hat{y}_n| & \text{if } y_n - \hat{y}_n \geq 0 \\
                    (1-\alpha) |y_n - \hat{y}_n| & \text{otherwise}
                \end{cases}
            $$
            """)

    def gradient_descent_():
        ''
        "## Gradient descent"
        cols = st.columns(2)
        cols[0].markdown(r"""
        
        Hyperparameter: learning rate. We want to use the biggest learning rate for which we dont fail.
        """)

        # lets draw a loss landscape
        x = np.linspace(-5,5,100)
        y = np.linspace(-5,5,100)
        X, Y = np.meshgrid(x,y)
        Z = X**2 + Y**2 + np.sin(X*8) 
        # add noise
        Z += np.random.normal(0, 3, Z.shape)
        # make smooth
        
        Z = gaussian_filter(Z, sigma=1) # to imoprt gaussian_filter: 

        
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_layout(title='Loss Landscape', autosize=False,
                            width=500, height=500,
                            margin=dict(l=15, r=15, b=15, t=35))
        cols[1].plotly_chart(fig)

    def test_train_split_():
        ''
        '## Test Train Split'
        cols = st.columns(2)
        cols[0].markdown(r"""
        
        Split data into training data, validation data, and test data. Thus we can notice if we have overfitted to out training data. simply use sklearn.model_selection.train_test_split.

        We should visualize of loss curves on the training and validation data. When the validation loss starts to increase, we have overfitted to the training data, and we should backtrack to the lowest validation loss. Then these parameters are the ones we should use for the test data.
        """)
        fig = plt.figure(figsize=(6,4))
        epochs = np.arange(0, 15)
        # training loss should be exponentially decreasing
        training_loss = np.exp(-epochs) + np.random.normal(0, .01, epochs.shape)
        offset = 1
        validation_loss = np.exp(-epochs) + offset + np.random.normal(0, .01, epochs.shape)

        plt.plot(epochs, training_loss, label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        cols[1].pyplot(fig)

        '### k-fold cross validation'
        """
        If we have a small dataset; we can emplot this method. 
        1. We split the data into k folds, 
        1. and then we train on k-1 folds, and validate on the remaining fold. 
        1. We repeat this k times, and then average the results. 

        This way we can use all the data for training and validation.
        """


        '### Cross Validation for time series'
        """
        If we have a time series, we should not use k-fold cross validation, because we will be using future data to predict past data. Instead we should use a sliding window.
        1. We split the data into k windows,
        1. and then we train on k-1 windows, and validate on the remaining window.
        1. We repeat this k times, and then average the results.

        """ 

    def decision_trees_():
        ''
        "## Decision Trees"
        cols = st.columns(2)
        with cols[0]:
            r"""
            Say we are dealing with a dataset with many missing values, a descision tree can handle this.
            
            ### Boosted descision trees (works great)
            * invariant under monotonic transformations (scaling, shifting)
            * robust to outliers, missing values, and irrelevant features
            * can handle mixed data types
            * works off-the-shelf (typically)

            A problem with a single tree,is overfitting! We can solve this by using a forest of trees, and then average the results. This is called a random forest.

            **Boosting** from adaboost
            $$
                y_\text{boost}(x) = \frac{1}{N} \sum_{n=1}^N \ln{\alpha_n} h_i(x)
            $$
            in which $\alpha$ is the boost rate $\alpha = \frac{1-\text{err}}{\text{err}}$, and $h_i$ is the weak learner (a single tree).

            The method focuses new trees on the examples that were misclassified by the previous trees. This is done by weighting the examples, and then training a new tree on the weighted examples. The weights are updated after each tree is trained. (This is called boosting)
            """
        with cols[1]:
            # lets try it on the XOR-dataset
            X = np.random.randint(0,2, (1000,2)).astype(float)
            y = np.logical_xor(X[:,0], X[:,1]).astype(float)            
            X += np.random.normal(0, .1, X.shape)*.5

            # test train split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

            fig, axes = plt.subplots(2,2, figsize=(6,6))
            axes[0,0].scatter(X_train[:,0], X_train[:,1], c=y_train, marker='x', label='train', alpha=.5)
            axes[0,0].scatter(X_test[:,0], X_test[:,1], c=y_test, marker='o', label='test', alpha=.5)

            axes[0,0].set_xlabel('x1')
            axes[0,0].set_ylabel('x2')
            axes[0,0].legend(loc='center')
            axes[0,0].set_title('XOR dataset (ground truth)')

            # import ada boost
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            #import xgboost as xgb
            from xgboost import XGBClassifier
            # import ligjhtgbm
            

            for i, classifier in enumerate([AdaBoostClassifier, DecisionTreeClassifier, XGBClassifier], start=1):
                ax = axes.flatten()
                y_pred = classifier().fit(X_train, y_train).predict(X_test)
                ax[i].scatter(X_train[:,0], X_train[:,1], c=y_train, marker='x', label='train', alpha=.5)
                ax[i].scatter(X_test[:,0], X_test[:,1], c=y_pred, marker='o', label='test', alpha=.5)
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                classifier_name = classifier.__name__
                ax[i].set_title(f'{classifier_name}')
                ax[i].legend(loc='center')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()


    loss_functions_()
    gradient_descent_()
    test_train_split_()
    decision_trees_()
        

    


    


if __name__ == '__main__':
    functions = [lecture_2,]
    with streamlit_analytics.track():
        navigator(functions)