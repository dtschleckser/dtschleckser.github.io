import React from 'react';
import { Collapse } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { FaArrowDown } from "react-icons/fa6";

const ExpandableText = ({ children, headerText }) => {
  const [opened, { toggle }] = useDisclosure(false);

  return (
    <div style={{overflowX: 'auto', wordWrap: 'break-word', maxWidth: '100%'}}>
      <h3 onClick={toggle} style={{ cursor: 'pointer', userSelect: 'none' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <FaArrowDown style={{
            transition: 'transform 0.2s',
            transform: `rotate(${opened ? 0 : '-90deg'})`
          }}/>
          <span style={{marginLeft: '0.5rem'}}>{headerText}</span>
        </div>
      </h3>
      <Collapse in={opened}>
        {children}
      </Collapse>
    </div>
  );
};

export default ExpandableText;