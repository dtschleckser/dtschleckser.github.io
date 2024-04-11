import moeHeaderArt from '../assets/blog-headers/moe-gradient.png'
import BlogListItem from '../components/BlogListItem/BlogListItem'

import { FaLinkedin, FaTwitter } from "react-icons/fa6";

function BlogRoot() {
    return <div>
        <h2>Posts</h2>
        <ul style={{ listStyleType: 'none', margin: '0px', padding: '0px' }}>
            <li>
                <BlogListItem
                    image={moeHeaderArt}
                    title="An Introduction to Sparsely Gated MoE"
                    subtitle="A primer on mixture of experts, sparse routing, related losses + architectures, and recent models"
                    date="April 4, 2024"
                    href="mixture-of-experts-intro"
                />
            </li>
        </ul>
        <div style={{position: 'fixed', left: '5px', bottom: '5px'}}>
            <a href="https://www.linkedin.com/in/daniel-schleckser/" style={{marginRight: '5px'}}>
                <FaLinkedin size={25}/>
            </a>
            <a href="https://www.twitter.com/dtschleckser">
                <FaTwitter size={25}/>
            </a>
        </div>
    </div>
}

export default BlogRoot