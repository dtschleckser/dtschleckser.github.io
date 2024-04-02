import './App.css'

import AltText from './components/AltText'
import ExpandableText from './components/ExpandableText'

import moeArch from './assets/moe-arch.png'
import mixtralExpertChoice from './assets/mixtral-expert-choice.png'
import denseVsMoe from './assets/dense-vs-moe.png'

import { InlineMath, BlockMath } from 'react-katex';

import { Code, Title } from '@mantine/core';

function App() {

  return (
    <div style={{textAlign: 'left', width: '100%', minWidth: '400px'}}>
      <Title order={1} style={{paddingBottom: '1rem', paddingTop: '2rem'}}>
        An Introduction to Sparsely Gated MoE
      </Title>
      <div>
        MoE (mixture of experts) models have experienced a significant increase in attention and popularity
        in recent months. Both open and closed-source LLMs have been trending towards sparsely
        gated mixture of expert models, with
        <AltText text="GPT-4" alt="There's very little official information about GPT-4's architecture.
        It's not confirmed to be a MoE, but NVIDIA's 2024 keynote featured a 'GPT-MoE-1.8T' model, 
        likely GPT-4." link="https://youtu.be/Y2F8yisiS6E?t=1193" /> and
        {' '}<a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1">Mixtral-8x7b</a>{' '}
        enjoying both high performance and fast inference relative to their total parameter count.
        So what does a "mixture of experts" do? How does it work and why would we want to use it? Hopefully, we'll
        make it to the end of this post clarifying all of that.
      </div>

      <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
        Sparsity
      </Title>
      <div>
        Parameter count is valuable in LLMs.
        {' '}<a href="https://arxiv.org/pdf/2001.08361.pdf">Neural scaling laws</a>{' '}
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
      such as NVIDIA's <a href="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/" target="_blank">Ampere architecture</a>,
      offer native support for sparse operations.
      </div>


      <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
        The Mixture of Experts Layer
      </Title>
      <div>
        The basic concept behind MoE has been around for a while, but a prominent example of its recent application is
        {' '}<a href="https://arxiv.org/pdf/1701.06538.pdf">The Sparsely-Gated Mixture of Experts Layer</a>.
        The authors introduce the concept of an MoE layer in the context of RNNs.
      </div>
      <br />
      <div>
        The MoE layer routes the input to a given number of
        "expert" networks, with each expert in Shazeer et al. being a 2-layer FFN with input of size 1024, output of size 512,
        and a ReLU activation function. Each of these experts gives their own unique answer, and we can choose how many experts we want to use
        (a hyperparameter usually called <Code>top_k_experts</Code>). We express the list of experts we're using as{' '}
        <InlineMath>{'\\{E_0, E_1, \\ldots, E_{n-1}\\}'}</InlineMath> when we have <InlineMath>n</InlineMath> total experts.
        <br />
        <br />
        Since we aren't using every parameter in every forward pass, we call the parameters that are actively being used "active parameters"
        and the total parameter count ("total" or "sparse" parameter count). 
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
      For each expert <InlineMath>E_i(x)</InlineMath> network, we determine whether
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
      (intuitively corresponding to the gating network's confidence in this expert). This also makes the gating network differentiable.
      Then, they're all summed and that's the output of the MoE layer!

      <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
        Modifications for Transformers
      </Title>
      In a Transformer model, we're dealing with transformer blocks that consist of a self-attention block and a FFN.
      When applying MoE to Transformers, we replace the standard single FFN in a Transformer block with a mixture of experts mechanism that has multiple expert FFNs.
      Similarly to the RNN case, we select the most appropriate one with a gating mechanism.
      <img src={denseVsMoe} style={{ width: 'inherit', paddingTop: '1rem', paddingBottom: '1rem' }}/>
      This figure is taken from <a href="https://arxiv.org/pdf/2209.01667.pdf">A Review of Sparse Expert Models in Deep Learning</a>.
      For a practical example, I'd recommend checking out the <a href="https://github.com/mistralai/mistral-src/blob/main/mistral/model.py#L145">Mixtral reference implementation</a>.

      <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
        How to Train Your Gating Network
      </Title>

      Choosing an appropriate architecture and loss to train our gating network is important, as we want
      to ensure that each expert ends up with a roughly even responsibility. We don't want a single expert that
      the gating network always routes to and receives all the data during training; this problem compounds during training,
      as the router will greatly prefer the expert that has seen more data. In all the below cases, our gating network
      is trained via backprop like the rest of the network, but there are some cases where the gating network isn't trained
      that way (e.g. <a href="https://arxiv.org/pdf/2106.04426.pdf">hash layers</a>). 

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

      <ExpandableText headerText="Importance Loss + Load Balancing Loss">
        Separately, Shazeer et al. add an auxiliary loss <InlineMath>{'L_{importance}'}</InlineMath>
        that's defined over a batch of data. They take the "importance value" of an expert, which is the sum
        of the values of the gating network in that batch:
        <div className='centeredMath'>
          <BlockMath>{'Importance(X) = \\sum_{x \\in X} G(x)'}</BlockMath>
        </div>
        Then, they add the squared <AltText text="coefficient of variation" alt="CV(X) = stdev(x) / mean(x)"
        link="https://en.wikipedia.org/wiki/Coefficient_of_variation#Definition" /> of the data, scaled by a
        hyperparameter <InlineMath>{'w_{importance}'}</InlineMath>:
        <div className='centeredMath'>
          <BlockMath>{'L_{importance}(X) = w_{importance} \\cdot CV(Importance(X))^2'}</BlockMath>
        </div>
        They also introduce a load balancing loss. It uses an estimator <InlineMath>{'Load(x)_i'}</InlineMath>, that
        is defined with data batch <InlineMath>{'X'}</InlineMath> for a single expert <InlineMath>{'i'}</InlineMath>:
        <div className='centeredMath'>
          <BlockMath>{'Load(X)_i = \\sum_{x \\in X} P(x, i)'}</BlockMath>
        </div>
        The exact definition for <InlineMath>{'P(x, i)'}</InlineMath> is ommitted here due to its verbosity, but the concept behind
        it is to use the probability that the gating function <InlineMath>{'G(x)_i'}</InlineMath> is non-zero for an expert i,
        given that we freeze the noise values for other experts. Keep in mind that our expert noise is determined by a
        learned scaling of a normal distribution, as defined in noisy top-k gating - we want to see if, after this noise, our
        expert <InlineMath>{'i'}</InlineMath> makes the cut for the top <InlineMath>{'k'}</InlineMath> experts.
        <br /><br />
        Once we have our load function defined, we can add the squared CV of it to the loss, adding a
        hyperparameter <InlineMath>{'w_{load}'}</InlineMath>, very similarly to our importance loss:
        <div className='centeredMath'>
          <BlockMath>{'L_{load}(X) = w_{load} \\cdot  CV(Load(X))^2'}</BlockMath>
        </div>
      </ExpandableText>

      <ExpandableText headerText="Switch Transformer Load Balancing Loss">
        <a href="https://arxiv.org/pdf/2101.03961.pdf">Switch transformers</a>{' '}
        greatly simplifies the load balancing + importance loss introduced by Shazeer et al. It introduces 
        an auxiliary loss to ensure that the load is balanced across experts. For:
        <ul>
          <li>Batch <InlineMath>{'\\beta'}</InlineMath></li>
          <li>Tokens <InlineMath>{'T'}</InlineMath> in the batch</li>
          <li>Experts <InlineMath>{'N'}</InlineMath> in the batch,</li>
        </ul>
        the auxiliary loss is defined by 
        <BlockMath>{'loss = \\alpha \\cdot N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i '}</BlockMath>
        where: <br />
        <InlineMath>{'f_i'}</InlineMath> is the fraction of tokens that get routed to expert i in the batch <InlineMath>{'\\beta'}</InlineMath>, and <br />
        <InlineMath>{'P_i'}</InlineMath> is the fraction of all router probability dedicated to expert i for all tokens in the batch <InlineMath>{'\\beta'}</InlineMath>.
        <br /> <br />
        Essentially, we want the router to have the token quantity and the routing probability to be even across all our experts.

      </ExpandableText>

      <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
        Open source MoE LLMs
      </Title>

      <div>
        A good example of a recent, high-performing, open-source MoE model is Mixtral. It has 46.7B total params, 12.9B of which
        are active (i.e. used during inference), and is highly competitive on benchmarks with non-MoE Transformer models like LLaMA 2 70B,
        while using 66% of the active parameter count. The primary drawback is the high VRAM cost, as all parameters need to be loaded
        in memory even if we're only using a subset - we're not likely to use all the same experts when generating even a couple tokens, so
        offloading experts wouldn't be efficient.
        <br />
        <br />
        The number of experts can vary greatly in open MoE LLMs. For example, the
        largest Switch Transformer model <a href="https://huggingface.co/google/switch-c-2048">switch-c-2048</a> has 2048 experts,
        resulting in a model with over a trillion active parameters. In comparison, Mixtral
        has <Code>top_k_experts = 2</Code> and <Code>n_experts = 8</Code>.
      </div>
      <h2>What do the experts do?</h2>
      Right now, it doesn't look like experts correspond cleanly to a single human-interpretable topic.
      They seem to focus more on token-level features: The below example, taken from the <a href="https://arxiv.org/pdf/2401.04088.pdf">Mixtral paper</a>,
      shows tokens such as commas and operators in Python being handled by one expert, while whitespace is handled by another.


      <img src={mixtralExpertChoice} style={{ width: 'inherit', paddingTop: '1rem', paddingBottom: '1rem' }}/>

      {
      // Mixtral of experts:
      // https://arxiv.org/abs/2401.04088

      // Sparsely gated MoE transformers:
      // https://arxiv.org/pdf/1701.06538.pdf
      }
      <br /><br />
    </div>
  )
}

export default App
