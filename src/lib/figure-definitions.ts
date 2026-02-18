export const FIGURE_DEFS: Record<string, { src: string; caption: string }> = {
  'mri-scan': {
    src: 'https://nci-media.cancer.gov/pdq/media/images/428431-571.jpg',
    caption: 'MRI example from medical imaging.',
  },
  'spect-scan': {
    src: 'https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20170104105121/ri/590/picture/Brain_SPECT_with_Acetazolamide_Slices_thumb.jpg',
    caption: 'SPECT scan example.',
  },
  'seismic-tomography': {
    src: 'http://www.earth.ox.ac.uk/~smachine/cgi/images/welcome_fig_tomo_depth.jpg',
    caption: 'Seismic tomography example.',
  },
  'claude-shannon': {
    src: 'https://d2r55xnwy6nx47.cloudfront.net/uploads/2020/12/Claude-Shannon_2880_Lede.jpg',
    caption: 'Claude Shannon, pioneer of information theory.',
  },
  'climate-grid': {
    src: 'https://caltech-prod.s3.amazonaws.com/main/images/TSchneider-GClimateModel-grid-LES-NEWS-WEB.width-450.jpg',
    caption: 'Model parameterization illustration.',
  },
  'gaussian-process': {
    src: 'https://gowrishankar.info/blog/gaussian-process-and-related-ideas-to-kick-start-bayesian-inference/gp.png',
    caption: 'Gaussian process visualization.',
  },
  'vertical-fault-diagram': {
    src: '/figures/vertical-fault-diagram.svg',
    caption: 'Simplified geometry used in vertical-fault inversion.',
  },
  'glacier-valley-diagram': {
    src: '/figures/glacier-valley-diagram.svg',
    caption: 'Glacier valley cross-section used in thickness inversion.',
  },
  'eigen-applications': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/35/Principal_component_analysis_of_Mars_data.png',
    caption: 'Eigenvalue applications in dimensionality reduction and modal analysis.',
  },
  'multiplicity-diagram': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Jordan_block.svg',
    caption: 'Algebraic vs geometric multiplicity and defective structure intuition.',
  },
  'convergence-comparison': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/8/8c/NewtonIteration_Ani.gif',
    caption: 'Qualitative convergence-rate comparison across iterative eigensolvers.',
  },
  'qr-convergence-plot': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/7e/QR_decomposition.svg',
    caption: 'QR-iteration intuition: repeated factorizations drive triangular convergence.',
  },
  'gershgorin-example': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/e/e2/Gerschgorin_disks.svg',
    caption: 'Gershgorin discs give fast eigenvalue location bounds in the complex plane.',
  },
  'algorithm-comparison': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/7a/Complexitysubsets.svg',
    caption: 'Algorithm trade-offs: convergence speed, robustness, and computational cost.',
  },
  'adl-mlops-loop': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/5/59/MLOps_lifecycle.png',
    caption: 'MLOps lifecycle diagram from data to deployment and monitoring.',
  },
  'adl-word2vec-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/01/Word2vec.png',
    caption: 'Skip-gram/Word2Vec architecture overview.',
  },
  'adl-lstm-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/9/93/LSTM_Cell.svg',
    caption: 'LSTM cell architecture and gating structure.',
  },
  'adl-gru-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/37/GRU.svg',
    caption: 'GRU architecture highlighting update and reset gates.',
  },
  'adl-cnn-feature-map': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png',
    caption: 'CNN feature hierarchy from local kernels to semantic representations.',
  },
  'continuum-stress-tensor': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/1/17/Stress_tensor_cartesian.svg',
    caption: 'Cartesian stress tensor components used in continuum mechanics.',
  },
  'continuum-hookes-law': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/4/47/Stress_strain_curve.svg',
    caption: 'Stress-strain relationship and Hooke-law linear regime.',
  },
  'continuum-density-fluctuations': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/73/Brownianmotion5cells.png',
    caption: 'Continuum approximation intuition from density fluctuations at finite scale.',
  },
  'complex-percolation-video': {
    src: 'https://archive.org/download/percolation-2d-simulation/percolation-2d-simulation.mp4',
    caption: 'Percolation evolution animation for cluster formation and critical threshold intuition.',
  },
  'complex-sandpile-image': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/0d/Sandpile_model.gif',
    caption: 'Self-organized criticality in sandpile avalanches.',
  },
  'orl-cartpole-learning-image': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/06/Cart-pole.svg',
    caption: 'CartPole control setup used for reinforcement learning experiments.',
  },
  'mdp-agent-environment-loop': {
    src: 'https://www.researchgate.net/publication/350130760/figure/fig2/AS:1002586224209929@1616046591580/The-agent-environment-interaction-in-MDP.png',
    caption: 'Agent-environment interaction loop for Markov decision processes.',
  },
};
