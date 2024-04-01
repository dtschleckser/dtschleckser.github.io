import './App.css'

import AltText from './components/AltText'
import ExpandableText from './components/ExpandableText'

import moeArch from './assets/moe-arch.png'
import mixtralExpertChoice from './assets/mixtral-expert-choice.png'

import { InlineMath, BlockMath } from 'react-katex';

import { Code } from '@mantine/core';

function App() {

  return (
    <div style={{textAlign: 'left', width: '100%', minWidth: '400px'}}>
      <h1>
        An Introduction to Sparsely Gated MoE
      </h1>
      <div>
        MoE (mixture of experts) models have experienced a significant increase in attention and popularity
        in recent months. Both open and closed-source LLMs have been trending towards sparsely
        gated mixture of expert models, with
        <AltText text="GPT-4" alt="There's very little official information about GPT-4's architecture.
        It's not confirmed to be a MoE, but NVIDIA's 2024 keynote featured a 'GPT-MoE-1.8T' model, 
        likely GPT-4." link="https://youtu.be/Y2F8yisiS6E?t=1193" />
        and <AltText text="Mixtral-8x7b" alt="Mixtral 8x7b on HuggingFace" link="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1" />
        enjoying both high performance and fast inference relative to their total parameter count.
        So what does a "mixture of experts" do? How does it work and why would we want to use it? Hopefully, we'll
        make it to the end of this post clarifying all of that.
      </div>
      <h2>
        Sparsity
      </h2>
      <div>
        Parameter count is valuable in LLMs.
        <AltText text="Neural scaling laws" alt="Kaplan et al., 2020" link="https://arxiv.org/pdf/2001.08361.pdf" />
        show us that we can significantly improve the performance of an LLM by scaling out, adding more and larger layers.
        However, it comes at a clear cost. The more parameters you have, the compute you need for both training and inference.
        We know that a neural network has parameters that are strongly activated by a particular input:
        What if we could only use the ones that are most relevant to our particular input? This is the
        observation and motivation behind sparsity in networks.
      </div>
      <br />
      <div>
      Sparsity allows us to selectively activate and compute only a subset of the model's parameters for each input,
      reducing the computational cost and memory footprint during inference. By leveraging sparsity, we can build
      larger models with more parameters while keeping the actual computation and memory usage manageable.
      </div>
      <br />
      <div>
      Recent advancements in hardware have made sparsity even more appealing. Modern GPUs and specialized AI accelerators,
      such as NVIDIA's <AltText text="Ampere architecture" alt="Explained in greater depth on the NVIDIA blog." link="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/" />,
      offer native support for sparse operations.
      </div>

      <h2>The Mixture of Experts Layer</h2>
      <div>
        The basic concept behind MoE has been around for a while, but a prominent example of its recent application is
        <AltText text="The Sparsely-Gated Mixture of Experts Layer" alt="Shazeer et al., 2017." link="https://arxiv.org/pdf/1701.06538.pdf" />.
        The authors introduce the concept of an MoE layer in the context of RNNs.
      </div>
      <br />
      <div>
        The MoE layer routes the input to a given number of
        "expert" networks, with each expert being a 2-layer FFN with input of size 1024, output of size 512, and a ReLU activation function.
        Each of these experts gives their own unique answer, and we can select how many experts we want to use
        (a hyperparameter usually called <Code>top_k_experts</Code>). We can express the list of experts we're using as{' '}
        <InlineMath>{'\\{E_0, E_1, \\ldots, E_{n-1}\\}'}</InlineMath>.
        <br />
        <br />
        To choose the experts we think will be best, we use a gating network - we'll call it <InlineMath>G(x)</InlineMath>.
        There are various ways of defining how <InlineMath>G(x)</InlineMath> will actually work, but for now we can assume
        it will black-box select the set of experts that are best to handle our input.
      </div>
      <br />
      To use our gating network to select experts, we'll use the formula:
      <div className='centeredMath'>
          <InlineMath>{'y = \\sum_{i=1}^{n} G(x)_i E_i(x)'}</InlineMath>
      </div>
      To break this down, for each expert <InlineMath>E_i(x)</InlineMath> network, we determine whether
      to use them with the gating network <InlineMath>G(x)_i</InlineMath>'s value at index <InlineMath>i</InlineMath>.
      If <InlineMath>G(x)_i</InlineMath> is 0, we don't need to calculate the value of the expert <InlineMath>E_i(x)</InlineMath> at all!
      Assuming we get <InlineMath>G(x)</InlineMath> into a nice distribution, we can select the top <InlineMath>k</InlineMath> experts,
      ensuring that we have a value for <InlineMath>k</InlineMath> that is low enough to be sparse but high enough to add enough information
      from our experts.
      <br /><br />
      With all this information in mind, here's the original MoE layer diagram from Shazeer et al.:

      <img src={moeArch} style={{ width: 'inherit' }}/>

      We can see our input <InlineMath>x</InlineMath> is being passed through a gating network and two experts are selected;
      they're being multiplied by the gating values <InlineMath>{'G(x)_2'}</InlineMath> and <InlineMath>{'G(x)_\{n-1\}'}</InlineMath> 
      (intuitively corresponding to the gating network's confidence in this expert).
      Then, they're all summed and that's the output of the MoE layer!

      <h2>Modifications for Transformers</h2>

      In a Transformer model, we're dealing with transformer blocks that consist of a self-attention block and a FFN.
      When applying MoE to Transformers, we use our MoE layer to have multiple expert FFNs and select the FFNs that are most appropriate.
      Similarly to the RNN case, each expert is a FFN and we select the most appropriate one with a gating mechanism.

      <h2>How to Train Your Gating Network</h2>

      Choosing an appropriate architecture and loss to train our gating network is really important here, as we want
      to ensure that each expert ends up with a roughly even responsibility. We don't want a single expert that
      the gating network always routes to and receives all the data during training; this problem compounds during training,
      as the router will greatly prefer the expert that has seen more data. In all the below cases, our gating network
      is trained via backprop like the rest of the network.

      <ExpandableText headerText="Softmax routing">
        A basic routing network that doesn't address this problem may look like a learnable matrix <InlineMath>{'W_g'}</InlineMath> into softmax:
        <div className='centeredMath'>
          <InlineMath>{'G(x) = Softmax(x \\cdot W_g)'}</InlineMath>
        </div>
        We take the input and multiply it with our learned matrix <InlineMath>{'W_g'}</InlineMath>, and then softmax to ensure we're able to
        scale evenly.
      </ExpandableText>

      <ExpandableText headerText="Noisy Top-K Gating">
        To start addressing the problem of load balancing our experts, we can
        add noise to our pre-softmax gating logits to even out the probability distribution
        and keep the top K outputs in order to make it sparse.
        <br /><br />
        First, the noise is added; it's scaled with a learnable matrix <InlineMath>{'W_\{noise\}'}</InlineMath>{' '}
        so that the amount of noise we add is appropriate for our given input:
        <div className='centeredMath'>
          <InlineMath>{'Noise(x) = StandardNormal() \\cdot Softplus(x \\cdot W_\{noise\})'}</InlineMath>
        </div>
        StandardNormal is just the standard normal distribution and Softplus is used to keep the outputs positive.
        We add the noise to the outputs of our gating network
        <div className='centeredMath'>
          <InlineMath>{'H(x) = (x \\cdot W_g) + Noise(x)'}</InlineMath>
        </div>
        After this, we can "KeepTopK" (simply keeping the greatest K values of x unmodified, with the rest set to -infinity).
        Then, we softmax the output of KeepTopK to get our selection of top K experts:
        <div className='centeredMath'>
          <InlineMath>{'G(x) = Softmax(KeepTopK(H(x), k))'}</InlineMath>
        </div>
      </ExpandableText>
      <ExpandableText headerText="Importance Loss">
        Separately, Shazeer et al. also add an additional loss <InlineMath>{'L_{importance}'}</InlineMath>
        that's defined over a batch of data. They take the "importance value" of an expert, which is the sum
        of the values of the gating network in that batch:
        <div className='centeredMath'>
          <BlockMath>{'Importance(X) = \\sum_{x \\in X} G(x)'}</BlockMath>
        </div>
        Then, they add the squared <AltText text="coefficient of variation" alt="CV(X) = stdev(x) / mean(x)"
        link="https://en.wikipedia.org/wiki/Coefficient_of_variation#Definition" /> of the data, scaled by a
        hyperparameter <InlineMath>{'w_{importance}'}</InlineMath>
        <div className='centeredMath'>
          <BlockMath>{'L_{importance}(X) = w_{importance} \\cdot CV(Importance(X))^2'}</BlockMath>
        </div>
      </ExpandableText>
      <ExpandableText headerText="Load-balancing Loss">
      </ExpandableText>
      <h2>Mixtral</h2>
      <h2>What do the experts do?</h2>
      A common misconception about "experts" in MoE is that they correspond to "experts" in easily human-interpretable topics 
      (i.e. one that specializes in biology or another that's an expert in Python). Interpretability research generally lags behind
      capability development, but right now, it doesn't look so cleanly interpretable. The following figure is from the Mixtral
      paper, with each token color-coded by its expert choice.

      <img src={mixtralExpertChoice} style={{ width: 'inherit', paddingTop: '1rem', paddingBottom: '1rem' }}/>

      {
      // Mixtral of experts:
      // https://arxiv.org/abs/2401.04088

      // Sparsely gated MoE transformers:
      // https://arxiv.org/pdf/1701.06538.pdf
      }
      <br /><br />
      "Active parameter" refers to a parameter
      that you actually have to use during a forward pass of the model - this isn't the case for all
      of them, which is why they're referred to as "sparse".
      - What?
      - Why?
      - What? (in depth)
      What are experts doing?
      - Examples
    </div>
  )
}

export default App
